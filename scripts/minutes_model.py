"""Step 3: minutes model.

Predicts how many minutes each player plays in a game, with a spread, given
that he plays. (Whether he plays at all is step 4.) Uses only information that
is known before tip-off: recent minutes, how often he starts, rest and
back-to-backs, which rotation players on his team are out, and a rough
blowout-risk proxy built from both teams' scoring margins so far.

Modes
  backtest  walk-forward test: every game is predicted using earlier games only.
            Trains on everything before 2026-01-15, checks on 2025-26 games from
            that date on, against simple baselines (last 10 games, season-to-date,
            exponential average). Writes data/minutes_backtest.json and
            data/minutes_params.json.
  project   fits on all games so far and projects every player on every upcoming
            game in data/schedule.csv. Writes data/minutes_projection.csv.
  both      backtest, then project.

Needs numpy. Reads data/player_games.csv, schedule.csv, injuries_latest.csv, and (when present)
injuries_official_latest.csv and play_probs.json from the official injury report step.
"""
import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from injury_report import DEFAULT_PLAY, last_first_key, norm_name

DATA = Path("data")
PLAYER_GAMES = DATA / "player_games.csv"
SCHEDULE = DATA / "schedule.csv"
INJURIES = DATA / "injuries_latest.csv"
PARAMS = DATA / "minutes_params.json"
BACKTEST = DATA / "minutes_backtest.json"
PROJECTION = DATA / "minutes_projection.csv"
OFFICIAL = DATA / "injuries_official_latest.csv"
PLAY_PROBS = DATA / "play_probs.json"

NBA = {
    "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU",
    "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL",
    "PHI", "PHX", "POR", "SAC", "SA", "TOR", "UTAH", "WSH",
}
VALID_TYPES = {"2", "3", "5"}
SPLIT_DATE = "2026-01-15"
TEST_SEASON = 2026
MIN_HIST = 5               # games of history needed to be scored in the backtest
HOME_EDGE = 2.5            # points, for the blowout proxy
LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
BIN_EDGES = [0, 10, 16, 22, 28, 33, 99]
QLEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
GRID = 41                  # points used to represent one residual distribution
MAX_UNCERTAIN = 6          # players per team whose play/sit is enumerated
SIMPLER_WINS_UNLESS = 0.002  # extra feature groups must cut squared error by 0.2% to be kept

FEATURES = [
    "l3", "l5", "l10", "ewma", "szn", "sd10", "start10", "last_start", "l10_x_start",
    "n_hist", "gap", "rest", "b2b", "g3", "home", "playoff",
    "abs_total", "abs_star", "abs_x", "blow", "blow_x",
]
GROUPS = {  # feature groups used for the ablation / parsimony test
    "absences": ["abs_total", "abs_star", "abs_x"],
    "blowout": ["blow", "blow_x"],
    "rest": ["rest", "b2b", "g3"],
}


def log(msg):
    print(msg, flush=True)


def ordinal(d):
    return datetime.fromisoformat(d).toordinal()


# ---------- loading ----------

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
                "date": r["game_date_et"], "event": r["event_id"], "sy": int(r["season_year"]),
                "stype": r["season_type"], "aid": r["athlete_id"], "name": r["player"],
                "team": r["team"], "opp": r["opp"], "home": int(r["home"]),
                "starter": int(r["starter"] or 0), "min": m, "pts": int(r["pts"] or 0),
            })
    rows.sort(key=lambda r: (r["date"], r["event"]))
    if not rows:
        raise SystemExit("no usable rows in data/player_games.csv")
    return rows


# ---------- state that is rolled forward game by game ----------

class State:
    def __init__(self):
        self.hist = defaultdict(lambda: deque(maxlen=10))   # aid -> (min, starter)
        self.ewma = {}
        self.n = defaultdict(int)
        self.szn = defaultdict(lambda: [0.0, 0])             # (aid, sy) -> [min, games]
        self.last_date = {}
        self.last_team = {}
        self.team_window = defaultdict(lambda: deque(maxlen=10))  # team -> {aid: min}
        self.team_season = {}
        self.team_dates = defaultdict(list)                  # team -> date ordinals
        self.margin = defaultdict(lambda: [0.0, 0])          # (sy, team) -> [sum diff, games]

    # --- team context ---
    def absentees(self, team, sy, played):
        """Rotation players of this team who are out: (total mpg, star count)."""
        if self.team_season.get(team) != sy:
            return 0.0, 0
        win = self.team_window[team]
        if not win:
            return 0.0, 0
        last = win[-1]
        count = defaultdict(list)
        for g in win:
            for aid, m in g.items():
                count[aid].append(m)
        total, stars = 0.0, 0
        for aid, ms in count.items():
            # "out" = rotation player who missed this game AND the previous team game.
            # Players who simply sit a night unannounced are not counted, because in real
            # use only announced absences (the injury list) are known before tip-off.
            if len(ms) < 2 or aid in played or aid in last or self.last_team.get(aid) != team:
                continue
            mpg = sum(ms) / len(ms)
            if mpg >= 8:
                total += mpg
                stars += mpg >= 24
        return total, stars

    def team_ctx(self, team, opp, home, d, sy):
        dates = self.team_dates[team]
        rest = min(d - dates[-1], 5) if dates else 5
        g3 = sum(1 for x in dates[-3:] if d - x <= 3)
        s, n = self.margin[(sy, team)]
        so, no = self.margin[(sy, opp)]
        em = s / (n + 5) - so / (no + 5) + (HOME_EDGE if home else -HOME_EDGE)
        return rest, g3, abs(em)

    # --- player features ---
    def features(self, aid, team, opp, home, date, sy, stype, absent):
        h = list(self.hist[aid])
        if not h:
            return None
        mins = [m for m, _ in h]
        d = ordinal(date)
        l10 = sum(mins) / len(mins)
        s = self.szn.get((aid, sy))
        szn = s[0] / s[1] if s and s[1] else l10
        sd10 = float(np.std(mins)) if len(mins) >= 3 else 5.0
        start10 = sum(st for _, st in h) / len(h)
        rest, g3, blow = self.team_ctx(team, opp, home, d, sy)
        abs_total, abs_star = absent
        gap = min(d - ordinal(self.last_date[aid]), 30) / 30.0
        return {
            "l3": sum(mins[-3:]) / len(mins[-3:]), "l5": sum(mins[-5:]) / len(mins[-5:]),
            "l10": l10, "ewma": self.ewma[aid], "szn": szn, "sd10": sd10,
            "start10": start10, "last_start": h[-1][1], "l10_x_start": l10 * start10,
            "n_hist": min(self.n[aid], 20), "gap": gap, "rest": rest, "b2b": float(rest == 1),
            "g3": g3, "home": home, "playoff": float(stype in ("3", "5")),
            "abs_total": abs_total, "abs_star": abs_star, "abs_x": abs_total * l10 / 36.0,
            "blow": blow, "blow_x": blow * l10 / 36.0,
        }

    def learn_day(self, day):
        by_event = defaultdict(lambda: defaultdict(list))
        for r in day:
            by_event[r["event"]][r["team"]].append(r)
        for ev, teams in by_event.items():
            pts = {t: sum(r["pts"] for r in rs) for t, rs in teams.items()}
            for t, rs in teams.items():
                sy, d = rs[0]["sy"], ordinal(rs[0]["date"])
                if self.team_season.get(t) != sy:
                    self.team_window[t].clear()
                    self.team_season[t] = sy
                self.team_window[t].append({r["aid"]: r["min"] for r in rs})
                self.team_dates[t].append(d)
                opp_pts = [v for k, v in pts.items() if k != t]
                if opp_pts:
                    m = self.margin[(sy, t)]
                    m[0] += pts[t] - opp_pts[0]
                    m[1] += 1
                for r in rs:
                    a = r["aid"]
                    self.hist[a].append((r["min"], r["starter"]))
                    self.ewma[a] = r["min"] if a not in self.ewma else self.ewma[a] + 0.2 * (r["min"] - self.ewma[a])
                    self.n[a] += 1
                    z = self.szn[(a, sy)]
                    z[0] += r["min"]
                    z[1] += 1
                    self.last_date[a] = r["date"]
                    self.last_team[a] = t


def walk(rows):
    """Chronological pass: features for every game, using only earlier games."""
    st = State()
    out = []
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["date"] == rows[i]["date"]:
            j += 1
        day = rows[i:j]
        played = defaultdict(set)
        for r in day:
            played[(r["event"], r["team"])].add(r["aid"])
        ab = {}
        for r in day:
            k = (r["event"], r["team"])
            if k not in ab:
                ab[k] = st.absentees(r["team"], r["sy"], played[k])
        for r in day:
            f = st.features(r["aid"], r["team"], r["opp"], r["home"], r["date"], r["sy"],
                            r["stype"], ab[(r["event"], r["team"])])
            if f is None:
                continue
            out.append({"x": f, "y": r["min"], "date": r["date"], "sy": r["sy"], "aid": r["aid"],
                        "team": r["team"], "n_hist": st.n[r["aid"]]})
        st.learn_day(day)
        i = j
    return out, st


# ---------- model ----------

def design(items, names):
    return np.array([[it["x"][n] for n in names] for it in items], dtype=float)


def fit_ridge(X, y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    ym = y.mean()
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - ym))
    return {"mu": mu, "sd": sd, "w": w, "b": ym}


def predict(m, X):
    return np.clip(((X - m["mu"]) / m["sd"]) @ m["w"] + m["b"], 0, 48)


def fit_best(train, names):
    """Pick the ridge penalty on the last 20% (by date) of the training rows, then refit on all."""
    train = sorted(train, key=lambda it: it["date"])
    cut = int(len(train) * 0.8)
    Xa, ya = design(train[:cut], names), np.array([t["y"] for t in train[:cut]])
    Xb, yb = design(train[cut:], names), np.array([t["y"] for t in train[cut:]])
    best = min(LAMBDAS, key=lambda lam: float(np.mean((predict(fit_ridge(Xa, ya, lam), Xb) - yb) ** 2)))
    X, y = design(train, names), np.array([t["y"] for t in train])
    return fit_ridge(X, y, best), best


def bin_of(p):
    for k in range(len(BIN_EDGES) - 1):
        if p < BIN_EDGES[k + 1]:
            return k
    return len(BIN_EDGES) - 2


def residual_quantiles(pred, y):
    res = y - pred
    bins = np.array([bin_of(p) for p in pred])
    overall = np.quantile(res, QLEVELS)
    out = []
    for k in range(len(BIN_EDGES) - 1):
        r = res[bins == k]
        out.append(np.quantile(r, QLEVELS) if len(r) >= 200 else overall)
    return np.array(out)


def interval(q, pred, lo, hi):
    ks = np.array([bin_of(p) for p in pred])
    li, hi_i = QLEVELS.index(lo), QLEVELS.index(hi)
    return pred + q[ks, li], pred + q[ks, hi_i]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def paired_z(pa, pb, y):
    d = (y - pa) ** 2 - (y - pb) ** 2
    se = d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean()), float(d.mean() / se) if se > 0 else float("nan")


def score(pred, y):
    return {"rmse": round(rmse(pred, y), 3), "mae": round(mae(pred, y), 3)}


# ---------- backtest ----------

def run_backtest(rows):
    feats, _ = walk(rows)
    train = [f for f in feats if f["date"] < SPLIT_DATE and f["n_hist"] >= MIN_HIST]
    valid = [f for f in feats if f["date"] >= SPLIT_DATE and f["sy"] == TEST_SEASON and f["n_hist"] >= MIN_HIST]
    log(f"  mean rotation minutes out per player-game in training: "
        f"{np.mean([f['x']['abs_total'] for f in train]):.1f}")
    log(f"Minutes backtest: {len(train)} training player-games, {len(valid)} validation player-games "
        f"(2025-26 from {SPLIT_DATE})")
    y = np.array([f["y"] for f in valid])

    # parsimony: start from all features, drop whole groups that do not clearly help
    names = list(FEATURES)
    model, lam = fit_best(train, names)
    full_mse = float(np.mean((predict(model, design(valid, names)) - y) ** 2))
    # use the training period (not validation) to decide which groups to keep
    tr_split = int(len(train) * 0.8)
    train_sorted = sorted(train, key=lambda it: it["date"])
    sub_a, sub_b = train_sorted[:tr_split], train_sorted[tr_split:]
    yb = np.array([t["y"] for t in sub_b])

    def holdout_mse(nms):
        m, _ = fit_best(sub_a, nms)
        return float(np.mean((predict(m, design(sub_b, nms)) - yb) ** 2))

    base_mse = holdout_mse(names)
    kept_groups, dropped = [], []
    for g, cols in GROUPS.items():
        without = [n for n in names if n not in cols]
        if holdout_mse(without) <= base_mse * (1 + SIMPLER_WINS_UNLESS):
            dropped.append(g)
        else:
            kept_groups.append(g)
    names = [n for n in FEATURES if not any(n in GROUPS[g] for g in dropped)]
    log(f"  feature groups kept: {kept_groups or 'none'}; dropped as not helping: {dropped or 'none'}")
    model, lam = fit_best(train, names)
    pred = predict(model, design(valid, names))

    base = {
        "last_10_avg": np.array([f["x"]["l10"] for f in valid]),
        "last_5_avg": np.array([f["x"]["l5"] for f in valid]),
        "exp_avg": np.array([f["x"]["ewma"] for f in valid]),
        "season_to_date": np.array([f["x"]["szn"] for f in valid]),
    }
    report = {"validation_rows": len(valid), "train_rows": len(train), "features_used": names,
              "groups_dropped": dropped, "ridge_lambda": lam, "model": score(pred, y)}
    report["baselines"] = {k: score(v, y) for k, v in base.items()}
    best_name = min(base, key=lambda k: rmse(base[k], y))
    d, z = paired_z(pred, base[best_name], y)
    report["vs_best_baseline"] = {"baseline": best_name, "mse_diff": round(d, 3), "z": round(z, 2)}

    # where does it help: by role and by whether rotation players are out
    l10 = np.array([f["x"]["l10"] for f in valid])
    abs_t = np.array([f["x"]["abs_total"] for f in valid])
    segs = {
        "starters (l10 >= 28)": l10 >= 28, "rotation (18-28)": (l10 >= 18) & (l10 < 28),
        "bench (< 18)": l10 < 18, "no rotation player out": abs_t == 0,
        "rotation player(s) out (>= 20 mpg)": abs_t >= 20,
        "back-to-back": np.array([f["x"]["b2b"] for f in valid]) == 1,
    }
    report["segments"] = {}
    for k, mask in segs.items():
        if mask.sum() >= 100:
            report["segments"][k] = {
                "rows": int(mask.sum()), "model": round(rmse(pred[mask], y[mask]), 3),
                "best_baseline": round(rmse(base[best_name][mask], y[mask]), 3)}

    # ablation on validation
    abl = {"full model": names}
    for g in GROUPS:
        abl[f"without {g}"] = [n for n in FEATURES if n not in GROUPS[g]]
    report["ablation_validation_rmse"] = {}
    for k, nms in abl.items():
        m, _ = fit_best(train, nms)
        report["ablation_validation_rmse"][k] = round(rmse(predict(m, design(valid, nms)), y), 3)

    # spread: residual quantiles from training rows, checked on validation
    tr_pred = predict(model, design(train, names))
    q = residual_quantiles(tr_pred, np.array([t["y"] for t in train]))
    lo10, hi90 = interval(q, pred, 0.10, 0.90)
    lo25, hi75 = interval(q, pred, 0.25, 0.75)
    report["coverage"] = {
        "80% interval (p10-p90) actually contains": round(float(np.mean((y >= lo10) & (y <= hi90))), 3),
        "50% interval (p25-p75) actually contains": round(float(np.mean((y >= lo25) & (y <= hi75))), 3),
        "below p10 (should be 0.10)": round(float(np.mean(y < lo10)), 3),
        "above p90 (should be 0.10)": round(float(np.mean(y > hi90)), 3),
    }
    cov_by_bin = {}
    ks = np.array([bin_of(p) for p in pred])
    for k in range(len(BIN_EDGES) - 1):
        mk = ks == k
        if mk.sum() >= 100:
            cov_by_bin[f"predicted {BIN_EDGES[k]}-{BIN_EDGES[k+1]} min"] = {
                "rows": int(mk.sum()),
                "p10-p90 coverage": round(float(np.mean((y[mk] >= lo10[mk]) & (y[mk] <= hi90[mk]))), 3)}
    report["coverage_by_predicted_minutes"] = cov_by_bin
    excluded = sum(1 for f in feats if f["date"] >= SPLIT_DATE and f["sy"] == TEST_SEASON and f["n_hist"] < MIN_HIST)
    report["validation_rows_skipped_for_short_history"] = excluded

    log("\nValidation (minutes, lower is better):")
    log(f"  {'method':<20}{'RMSE':>8}{'MAE':>8}")
    log(f"  {'model':<20}{report['model']['rmse']:>8}{report['model']['mae']:>8}")
    for k, v in report["baselines"].items():
        log(f"  {k:<20}{v['rmse']:>8}{v['mae']:>8}")
    log(f"  model vs {best_name}: z = {report['vs_best_baseline']['z']}")
    log("\nBy segment (RMSE model vs best baseline):")
    for k, v in report["segments"].items():
        log(f"  {k:<38} rows={v['rows']:<6} model={v['model']}  baseline={v['best_baseline']}")
    log("\nAblation (validation RMSE):")
    for k, v in report["ablation_validation_rmse"].items():
        log(f"  {k:<24}{v}")
    log("\nSpread check (are the intervals honest?):")
    for k, v in report["coverage"].items():
        log(f"  {k}: {v}")
    for k, v in cov_by_bin.items():
        log(f"  {k}: rows={v['rows']} p10-p90 coverage={v['p10-p90 coverage']}")
    log(f"  validation rows skipped (under {MIN_HIST} games of history): {excluded}")

    DATA.mkdir(exist_ok=True)
    BACKTEST.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return names


# ---------- projection ----------

def run_project(rows, names):
    feats, st = walk(rows)
    usable = [f for f in feats if f["n_hist"] >= MIN_HIST]
    model, lam = fit_best(usable, names)
    tr_pred = predict(model, design(usable, names))
    q = residual_quantiles(tr_pred, np.array([t["y"] for t in usable]))
    PARAMS.write_text(json.dumps({
        "features": names, "ridge_lambda": lam, "weights": [round(float(w), 5) for w in model["w"]],
        "intercept": round(float(model["b"]), 4), "bin_edges": BIN_EDGES, "quantile_levels": QLEVELS,
        "residual_quantiles": [[round(float(v), 3) for v in row] for row in q],
        "trained_rows": len(usable), "trained_through": rows[-1]["date"],
        "fit_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    if not SCHEDULE.exists():
        raise SystemExit("data/schedule.csv missing")
    today_et = datetime.now(timezone.utc).astimezone(_et()).date().isoformat()
    games = []
    with open(SCHEDULE, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["state"] == "pre" and r["game_date_et"] >= today_et and r["home"] in NBA and r["away"] in NBA:
                games.append(r)
    games.sort(key=lambda r: (r["game_date_et"], r["event_id"]))

    out_status = {}
    if INJURIES.exists():
        with open(INJURIES, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out_status[r["athlete_id"]] = r["status"]
    official = defaultdict(dict)          # (game_date, team) -> {player_norm: status}
    if OFFICIAL.exists():
        with open(OFFICIAL, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                official[(r["game_date_et"], r["team"])][r["player_norm"]] = r["status"]
    play_probs, probs_source = dict(DEFAULT_PLAY), "defaults"
    if PLAY_PROBS.exists():
        pp = json.loads(PLAY_PROBS.read_text(encoding="utf-8"))
        play_probs.update(pp.get("probs", {}))
        probs_source = pp.get("source", "defaults")
    names_by_id = {r["aid"]: r["name"] for r in rows}
    sy_now = rows[-1]["sy"]
    # residual shapes by predicted-minutes bin, as 41 evenly spaced quantile points
    tr_res = np.array([t["y"] for t in usable]) - tr_pred
    tr_bins = np.array([bin_of(p) for p in tr_pred])
    grid_levels = (np.arange(GRID) + 0.5) / GRID
    res_grid = []
    for k in range(len(BIN_EDGES) - 1):
        r = tr_res[tr_bins == k]
        res_grid.append(np.quantile(r if len(r) >= 200 else tr_res, grid_levels))
    res_grid = np.array(res_grid)

    out = []
    stats = Counter()
    for g in games:
        d = ordinal(g["game_date_et"])
        for team, opp, home in ((g["home"], g["away"], 1), (g["away"], g["home"], 0)):
            window = st.team_window[team] if st.team_season.get(team) == sy_now else []
            seen = defaultdict(list)
            for gm in window:
                for aid, m in gm.items():
                    seen[aid].append(m)
            cands = {a: ms for a, ms in seen.items() if st.last_team.get(a) == team and len(ms) >= 2}
            mpg = {a: sum(ms) / len(ms) for a, ms in cands.items()}
            off = official.get((g["game_date_et"], team))
            norm_of = {a: norm_name(names_by_id.get(a, "")) for a in cands}
            lf_count = Counter(last_first_key(n) for n in norm_of.values())
            p_play, status, source = {}, {}, {}
            for a in cands:
                stt, src = None, "none"
                if off:
                    stt = off.get(norm_of[a])
                    if stt is None:  # fall back to last name + first initial when unique on the team
                        lf = last_first_key(norm_of[a])
                        hit = [v for k2, v in off.items() if last_first_key(k2) == lf]
                        if lf_count[lf] == 1 and len(hit) == 1:
                            stt = hit[0]
                    if stt is not None:
                        src = "official"
                    else:
                        stt, src = "", "official_not_listed"
                e = out_status.get(a, "")
                if not stt and e == "Out":  # ESPN says out and the official report does not contradict it
                    stt, src = "Out", "espn"
                elif not stt and e and src == "none":  # no official report for this game yet
                    stt, src = "Questionable", "espn"  # ESPN "Day-To-Day" has no finer level
                status[a], source[a] = stt or "", src
                p_play[a] = play_probs.get(stt, 1.0) if stt else 1.0
            sure_out = [a for a in cands if p_play[a] <= 0.03]
            unsure = sorted((a for a in cands if 0.03 < p_play[a] < 0.97), key=lambda a: -mpg[a])
            branch = unsure[:MAX_UNCERTAIN]
            fixed_out = set(sure_out) | {a for a in unsure[MAX_UNCERTAIN:] if p_play[a] < 0.5}
            active_always = [a for a in cands if a not in fixed_out and a not in branch]
            # one base feature row per player; only the absence features change by scenario
            base = {}
            for a in active_always + branch:
                f = st.features(a, team, opp, home, g["game_date_et"], sy_now, g["season_type"], (0.0, 0))
                if f is not None:
                    base[a] = f
            pts = defaultdict(list)       # aid -> [(weight, minutes)]
            wsum = defaultdict(float)
            for combo in range(2 ** len(branch)):
                w, outs = 1.0, set(fixed_out)
                for i, a in enumerate(branch):
                    if combo >> i & 1:
                        w *= p_play[a]
                    else:
                        w *= 1 - p_play[a]
                        outs.add(a)
                if w < 1e-6:
                    continue
                tot = sum(mpg[a] for a in outs if mpg[a] >= 8)
                stars = sum(1 for a in outs if mpg[a] >= 24)
                playing = [a for a in base if a not in outs]
                if not playing:
                    continue
                X = []
                for a in playing:
                    f = dict(base[a])
                    f["abs_total"], f["abs_star"], f["abs_x"] = tot, stars, tot * f["l10"] / 36.0
                    X.append([f[n] for n in names])
                mu = predict(model, np.array(X, dtype=float))
                for a, m in zip(playing, mu):
                    wsum[a] += w
                    pts[a].append((w, float(m), bin_of(m)))
            for a, lst in pts.items():
                tw = sum(w for w, _, _ in lst)
                mean = sum(w * m for w, m, _ in lst) / tw
                vals, wts = [], []
                for w, m, k in lst:
                    vals.extend(np.clip(m + res_grid[k], 0, 60))
                    wts.extend([w / GRID] * GRID)
                order = np.argsort(vals)
                vals, wts = np.array(vals)[order], np.array(wts)[order]
                cdf = np.cumsum(wts) / wts.sum()
                qv = {lvl: float(vals[min(np.searchsorted(cdf, lvl), len(vals) - 1)]) for lvl in QLEVELS}
                pp_a = p_play[a]
                row = {"game_date_et": g["game_date_et"], "event_id": g["event_id"], "team": team,
                       "opp": opp, "home": home, "athlete_id": a, "player": names_by_id.get(a, ""),
                       "status": status[a], "status_source": source[a], "p_play": round(pp_a, 3),
                       "proj_min": round(mean, 1), "exp_min": round(pp_a * mean, 1)}
                for lvl, nm in zip(QLEVELS, ("p05", "p10", "p25", "p50", "p75", "p90", "p95")):
                    row[nm] = round(qv[lvl], 1)
                f = base[a]
                row.update({"last10_avg": round(f["l10"], 1), "starter_share_l10": round(f["start10"], 2),
                            "team_rotation_min_out": round(sum(mpg[x] for x in cands
                                                               if p_play[x] <= 0.03 and mpg[x] >= 8), 1),
                            "rest_days": f["rest"], "games_of_history": st.n[a],
                            "low_history": int(st.n[a] < MIN_HIST)})
                stats["low_history"] += row["low_history"]
                stats["src_" + source[a]] += 1
                out.append(row)
        for team in (g["home"], g["away"]):
            st.team_dates[team].append(d)
    if not out:
        log("No upcoming games to project (schedule empty or all games started).")
    else:
        out.sort(key=lambda r: (r["game_date_et"], r["event_id"], r["team"], -r["exp_min"]))
        with open(PROJECTION, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
        log(f"  mean rotation minutes out per projected player-game: "
            f"{np.mean([r['team_rotation_min_out'] for r in out]):.1f}")
    log(f"\nProjected {len(out)} player-games across {len(games)} upcoming games "
        f"(as of {today_et} ET); {stats['low_history']} flagged low-history. "
        f"Status source: official={stats['src_official']}, official-but-not-listed={stats['src_official_not_listed']}, "
        f"espn={stats['src_espn']}, none={stats['src_none']}. Play probabilities: {probs_source}. "
        f"Wrote {PROJECTION}")
    return out


def _et():
    from zoneinfo import ZoneInfo
    return ZoneInfo("America/New_York")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["backtest", "project", "both"], required=True)
    args = p.parse_args(argv)
    rows = load_rows()
    log(f"Loaded {len(rows)} usable player-games")
    names = list(FEATURES)
    if args.mode in ("backtest", "both"):
        names = run_backtest(rows)
    elif PARAMS.exists():
        names = json.loads(PARAMS.read_text(encoding="utf-8"))["features"]
    if args.mode in ("project", "both"):
        run_project(rows, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
