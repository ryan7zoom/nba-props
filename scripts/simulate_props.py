"""Step 5: distributions and PRA simulation.

For every projected player-game this turns the earlier steps into a full
distribution of points, rebounds, assists and PRA (points + rebounds + assists),
GIVEN THAT THE PLAYER PLAYS. Sportsbooks normally void a prop when the player does
not play, so p_play is reported next to the numbers rather than mixed into them.

How one draw is made (default 20,000 draws per player-game)
  1. Minutes come from the step 3/4 minutes distribution (p05..p95, scenario-mixed).
  2. Points / rebounds / assists are overdispersed counts (gamma-Poisson) with mean
     rate_per_minute x minutes, using the step 2 rates and their phi.
  3. Points get a minutes coupling: output per minute is lower when a player plays
     far fewer minutes than expected (early exits, garbage time) and a little higher
     when he plays a lot. The exponent is fitted on past games, and kept only if it
     clearly improves the fit.
  4. A small share of games are "zero-points" games. Also fitted and kept only if it
     clearly improves the fit.
  Points, rebounds and assists share the same minutes draw, which is what correlates
  them. The backtest checks that this is enough (leftover correlation is tiny).

Modes
  backtest  fit on 2025-26 before 15/01/2026, check on the rest of 2025-26 (no leakage:
            each game uses only information from before it). Reports accuracy vs simple
            baselines, calibration, and which pieces earn their keep. Writes
            data/sim_backtest.json.
  project   refit the small set of simulation settings on all games so far, then
            simulate every player-game in data/minutes_projection.csv. Writes
            data/projections.csv, data/projections_latest.json, data/sim_params.json,
            data/sim_health.json. Exits with an error if anything is wrong.
  both      backtest, then project.

Needs numpy. Reads the files made by update_data.py, baseline_rates.py and minutes_model.py.
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import baseline_rates as br
import minutes_model as mm
from injury_report import dmy, dmy_time

DATA = Path("data")
PLAYER_GAMES = DATA / "player_games.csv"
MIN_PROJECTION = DATA / "minutes_projection.csv"
PLAYER_RATES = DATA / "player_rates.csv"
SIM_PARAMS = DATA / "sim_params.json"
SIM_BACKTEST = DATA / "sim_backtest.json"
OUT_CSV = DATA / "projections.csv"
OUT_JSON = DATA / "projections_latest.json"
HEALTH = DATA / "sim_health.json"

SPLIT_DATE = mm.SPLIT_DATE          # 15/01/2026: fit before, check after
TEST_SEASON = mm.TEST_SEASON        # season_year 2026 = 2025-26
N_BACKTEST = 3000
N_PROJECT = 20000
SEED = 20261026
STATS = ("pts", "reb", "ast")

KU = np.array([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0])   # minutes quantile knots
MIN_CAP = 53.0
DELTA_GRID = [round(float(x), 2) for x in np.arange(-0.10, 0.481, 0.02)]
PHI_GRID = [round(float(x), 2) for x in np.arange(1.0, 3.31, 0.1)]
PI_GRID = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]
COUPLING_MIN_GAIN = 0.003           # exponent kept only if it cuts the Poisson deviance by 0.3%
ZERO_MIN_LL_GAIN = 10.0             # zero-points share kept only if log-likelihood rises by 10
RHO_FLOOR = 0.02
MIN_FIT_ROWS = 1500
REQUIRED_COLS = ["game_date_et", "event_id", "team", "opp", "home", "athlete_id", "player", "status",
                 "status_source", "p_play", "proj_min", "p05", "p10", "p25", "p50", "p75", "p90", "p95"]
LINE_OFFSETS = (0.0, 0.35, -0.35)   # test lines: near the median, and a bit above / below it
CDF_DONE = 0.9999


def log(msg=""):
    print(msg, flush=True)


def _et():
    from zoneinfo import ZoneInfo
    return ZoneInfo("America/New_York")


# ---------- loading ----------

def load_rows():
    """Every game a player actually played (minutes > 0), oldest first."""
    rows = []
    with open(PLAYER_GAMES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["team"] not in mm.NBA or r["opp"] not in mm.NBA:
                continue
            if r["season_type"] not in mm.VALID_TYPES or r["dnp"] == "1":
                continue
            m = float(r["min"])
            if m <= 0:
                continue
            pts, reb, ast = int(r["pts"] or 0), int(r["reb"] or 0), int(r["ast"] or 0)
            rows.append({
                "date": r["game_date_et"], "event": r["event_id"], "sy": int(r["season_year"]),
                "stype": r["season_type"], "aid": r["athlete_id"], "name": r["player"],
                "team": r["team"], "opp": r["opp"], "home": int(r["home"]),
                "starter": int(r["starter"] or 0), "min": m, "pts": pts,
                "pos": br.POS_MAP.get(r["position"], "UNK"), "y": (pts, reb, ast),
            })
    rows.sort(key=lambda r: (r["date"], r["event"]))
    if not rows:
        raise SystemExit("no usable rows in data/player_games.csv")
    return rows


def rate_snapshots(rows, rp, seasons):
    """{(date, athlete_id): per-minute rates (pts, reb, ast)} as known BEFORE each game."""
    old = br.MIN_EVAL
    br.MIN_EVAL = 0.0                 # score every game he played, not only 10+ minute games
    out = {}
    try:
        have = {r["sy"] for r in rows}
        for target in sorted(seasons):
            if target - 1 not in have:
                continue
            for f in br.build_features(rows, target):
                rates = []
                for s, name in enumerate(STATS):
                    p = rp[name]
                    rates.append(br.blended_rate(f, s, p["variant"], p["K"], p["a"], rp["b"], p["c"]))
                out[(f["date"], f["aid"])] = tuple(rates)
    finally:
        br.MIN_EVAL = old
    return out


def minutes_setup(feats, names):
    """Fit the step 3 ridge on rows before the split (backtest) or all rows (project)."""
    return mm.fit_best(feats, names)


def make_items(feats_sel, pred, qtab, rmap):
    keep = [(f, p) for f, p in zip(feats_sel, pred) if (f["date"], f["aid"]) in rmap]
    return {
        "q7": np.array([p + qtab[mm.bin_of(p)] for f, p in keep], dtype=float).reshape(-1, 7),
        "mean": np.array([p for f, p in keep], dtype=float),
        "rates": np.array([rmap[(f["date"], f["aid"])] for f, p in keep], dtype=float).reshape(-1, 3),
        "m": np.array([f["y"] for f, p in keep], dtype=float),
        "team": [f["team"] for f, p in keep],
        "date": [f["date"] for f, p in keep],
        "aid": [f["aid"] for f, p in keep],
    }


def attach_actuals(it, row_lookup):
    it["y"] = np.array([row_lookup[(d, a)]["y"] for d, a in zip(it["date"], it["aid"])], dtype=float).reshape(-1, 3)
    return it


# ---------- the simulation ----------

def minute_knots(q7, mean):
    """Quantiles at 5,10,25,50,75,90,95% -> 9 knot values (0% and 100% extrapolated), mean-matched."""
    q7 = np.maximum.accumulate(np.clip(np.asarray(q7, dtype=float), 0, MIN_CAP), axis=1)
    lo = np.maximum(0.0, q7[:, 0] - 2.0 * (q7[:, 1] - q7[:, 0]))
    hi = np.minimum(MIN_CAP, q7[:, 6] + 2.0 * (q7[:, 6] - q7[:, 5]))
    v = np.concatenate([lo[:, None], q7, hi[:, None]], axis=1)
    implied = ((v[:, :-1] + v[:, 1:]) / 2.0 * np.diff(KU)).sum(1)
    v = np.clip(v + (np.asarray(mean, dtype=float) - implied)[:, None], 0, MIN_CAP)
    return np.maximum.accumulate(v, axis=1)


def draw_minutes(v, u):
    j = np.clip(np.searchsorted(KU, u, side="right") - 1, 0, len(KU) - 2)
    u0, u1 = KU[j], KU[j + 1]
    v0 = np.take_along_axis(v, j, axis=1)
    v1 = np.take_along_axis(v, j + 1, axis=1)
    return v0 + (u - u0) / (u1 - u0) * (v1 - v0)


def nb_counts(mu, phi, rng):
    """Counts with mean mu and variance phi*mu (gamma-Poisson); plain Poisson when phi is ~1."""
    mu = np.maximum(mu, 1e-6)
    if phi <= 1.005:
        return rng.poisson(mu)
    return rng.poisson(rng.gamma(mu / (phi - 1.0), phi - 1.0))


def simulate(q7, mean_min, rates, P, n, rng, point_minutes=False):
    """Draws for a batch of player-games. Returns pts, reb, ast, pra, min as (batch, n) arrays."""
    mean_min = np.asarray(mean_min, dtype=float)
    if point_minutes:
        m = np.repeat(mean_min[:, None], n, axis=1)
    else:
        m = draw_minutes(minute_knots(q7, mean_min), rng.random((len(mean_min), n)))
    rho = np.maximum(m / np.maximum(mean_min[:, None], 1.0), RHO_FLOOR)
    out = {}
    for s, name in enumerate(STATS):
        mu = P["c"][s] * rates[:, s][:, None] * m
        if P["delta"][s]:
            mu = mu * rho ** P["delta"][s]
        if name == "pts" and P["pi_pts"] > 0:
            cnt = nb_counts(mu / (1.0 - P["pi_pts"]), P["phi"][s], rng)
            cnt[rng.random(cnt.shape) < P["pi_pts"]] = 0
        else:
            cnt = nb_counts(mu, P["phi"][s], rng)
        out[name] = cnt
    out["pra"] = out["pts"] + out["reb"] + out["ast"]
    out["min"] = m
    return out


# ---------- fitting the settings ----------

_lg = np.frompyfunc(math.lgamma, 1, 1)


def _lgamma(x):
    return np.array(_lg(x), dtype=float)


def nb_logpmf(y, mu, phi, lg_y1):
    """Log probability of count y; lg_y1 = log(y!) precomputed once."""
    mu = np.maximum(mu, 1e-6)
    r = mu / (phi - 1.0)
    p = 1.0 / phi
    return _lgamma(y + r) - _lgamma(r) - lg_y1 + r * math.log(p) + y * math.log1p(-p)


def poisson_deviance(y, mu):
    mu = np.maximum(mu, 1e-9)
    return float(2.0 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0.0) - (y - mu)))


def fit_params(it, no_coupling=False, no_zero=False):
    """Small set of settings, fitted on past games with actual minutes known."""
    y, r, m, mean = it["y"], it["rates"], it["m"], it["mean"]
    rho = np.maximum(m / np.maximum(mean, 1.0), RHO_FLOOR)
    delta, c, phi, notes = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], {}
    for s, name in enumerate(STATS):
        def dev_at(dl):
            base = r[:, s] * m * rho ** dl
            cc = y[:, s].sum() / base.sum()
            return poisson_deviance(y[:, s], cc * base), cc
        d0, c0 = dev_at(0.0)
        best_d, best_dl, best_c = d0, 0.0, c0
        if not no_coupling:
            for dl in DELTA_GRID:
                d, cc = dev_at(dl)
                if d < best_d:
                    best_d, best_dl, best_c = d, dl, cc
        gain = (d0 - best_d) / d0 if d0 > 0 else 0.0
        keep = (not no_coupling) and gain >= COUPLING_MIN_GAIN and best_dl != 0.0
        delta[s] = best_dl if keep else 0.0
        c[s] = best_c if keep else c0
        notes[name + "_coupling"] = {"best_exponent": best_dl, "deviance_gain": round(gain, 5), "kept": bool(keep)}
    mu = [c[s] * r[:, s] * m * rho ** delta[s] for s in range(3)]
    for s in (1, 2):   # rebounds, assists: moment estimate of overdispersion
        phi[s] = max(1.0, float(np.sum((y[:, s] - mu[s]) ** 2) / np.sum(mu[s])))
    # points: overdispersion and zero-points share together, by likelihood
    best = {}
    lg_y1 = _lgamma(y[:, 0] + 1.0)
    for pi in ([0.0] if no_zero else PI_GRID):
        norm = 1.0 / (1.0 - pi)
        for ph in PHI_GRID[1:]:
            la = nb_logpmf(y[:, 0], mu[0] * norm, ph, lg_y1)
            if pi > 0:
                lz = np.where(y[:, 0] == 0, 0.0, -np.inf)
                ll = float(np.sum(np.logaddexp(math.log(1 - pi) + la, math.log(pi) + lz)))
            else:
                ll = float(np.sum(la))
            if pi not in best or ll > best[pi][0]:
                best[pi] = (ll, ph)
    ll0, ph0 = best[0.0]
    pi_best = max(best, key=lambda k: best[k][0])
    gain_ll = best[pi_best][0] - ll0
    if pi_best > 0 and gain_ll >= ZERO_MIN_LL_GAIN:
        pi_pts, phi[0] = pi_best, best[pi_best][1]
    else:
        pi_pts, phi[0] = 0.0, ph0
    notes["pts_zero_share"] = {"best_share": pi_best, "loglik_gain": round(gain_ll, 1), "kept": pi_pts > 0}
    return {"delta": delta, "c": [round(x, 4) for x in c], "phi": [round(float(x), 3) for x in phi],
            "pi_pts": pi_pts, "fit_rows": int(len(m)), "notes": notes}


# ---------- baselines used in the backtest ----------

def baseline_table(rows, keys):
    """For each wanted (date, athlete_id): last <=20 games' (pts, reb, ast) and season-to-date averages."""
    want = set(keys)
    hist = defaultdict(lambda: deque(maxlen=20))
    szn = defaultdict(lambda: [0, 0, 0, 0])
    out = {}
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["date"] == rows[i]["date"]:
            j += 1
        for r in rows[i:j]:
            k = (r["date"], r["aid"])
            if k in want:
                s = szn[(r["aid"], r["sy"])]
                out[k] = (list(hist[r["aid"]]), tuple(s[q] / s[3] for q in range(3)) if s[3] else None)
        for r in rows[i:j]:
            hist[r["aid"]].append(r["y"])
            s = szn[(r["aid"], r["sy"])]
            for q in range(3):
                s[q] += r["y"][q]
            s[3] += 1
        i = j
    return out


# ---------- evaluation ----------

def evaluate(it, P, n, seed, point_minutes=False, bs=400, fixed_lines=None):
    """Simulate every row and score it. Arrays are returned per row, for paired comparisons.
    fixed_lines = {"pts": (rows, 3), "pra": (rows, 3)} makes every variant answer the same questions."""
    rng = np.random.default_rng(seed)
    res = {s: {k: [] for k in ("mean", "med", "sd", "crps", "pit", "lp", "lo", "lines")} for s in ("pts", "pra")}
    idx = np.arange(1, n + 1)
    for a in range(0, len(it["mean"]), bs):
        sl = slice(a, a + bs)
        d = simulate(it["q7"][sl], it["mean"][sl], it["rates"][sl], P, n, rng, point_minutes)
        y3 = it["y"][sl]
        for s, col in (("pts", 0), ("pra", 3)):
            x = d[s].astype(float)
            y = y3[:, 0] if s == "pts" else y3.sum(1)
            xs = np.sort(x, axis=1)
            o = res[s]
            o["mean"].append(x.mean(1))
            med = np.median(x, axis=1)
            sd = x.std(1)
            o["med"].append(med)
            o["sd"].append(sd)
            e2 = (2.0 / n ** 2) * ((2 * idx - n - 1)[None, :] * xs).sum(1)
            o["crps"].append(np.abs(x - y[:, None]).mean(1) - 0.5 * e2)
            fy, fm = (x <= y[:, None]).mean(1), (x < y[:, None]).mean(1)
            o["pit"].append(fm + rng.random(len(y)) * (fy - fm))
            if fixed_lines is not None:
                lines = fixed_lines[s][sl]
            else:
                lines = np.stack([np.floor(med + k * sd) + 0.5 for k in LINE_OFFSETS], axis=1)
            o["lp"].append(np.stack([(x > lines[:, q][:, None]).mean(1) for q in range(3)], axis=1))
            o["lo"].append((y[:, None] > lines).astype(float))
            o["lines"].append(lines)
    for s in res:
        for k in res[s]:
            res[s][k] = np.concatenate(res[s][k])
    return res


def paired(a, b):
    """Mean of (a - b) per row and its z-score; negative = a is lower (better for losses)."""
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean()), float(d.mean() / se) if se > 0 else float("nan")


def brier_rows(p, o):
    return ((p - o) ** 2).mean(1)


def reliability(p, o, bins=10):
    p, o = p.ravel(), o.ravel()
    out, ece = [], 0.0
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        mk = (p >= lo) & (p < hi) if k < bins - 1 else (p >= lo) & (p <= hi)
        if mk.sum() >= 30:
            out.append({"predicted_range": f"{int(lo * 100)}-{int(hi * 100)}%", "n": int(mk.sum()),
                        "mean_predicted": round(float(p[mk].mean()), 3), "actually_over": round(float(o[mk].mean()), 3)})
            ece += mk.sum() / len(p) * abs(float(o[mk].mean()) - float(p[mk].mean()))
    return out, round(ece, 4)


def coverage(pit):
    return {"p10_p90": round(float(np.mean((pit > 0.1) & (pit < 0.9))), 3),
            "p25_p75": round(float(np.mean((pit > 0.25) & (pit < 0.75))), 3),
            "below_p10": round(float(np.mean(pit < 0.1)), 3), "above_p90": round(float(np.mean(pit > 0.9)), 3)}


# ---------- backtest ----------

def run_backtest(rows):
    t0 = time.time()
    rp, _ = br.load_params()
    names = json.loads(mm.PARAMS.read_text(encoding="utf-8"))["features"] if mm.PARAMS.exists() else list(mm.FEATURES)
    feats, _ = mm.walk(rows)
    rmap = rate_snapshots(rows, rp, [TEST_SEASON])
    train_f = [f for f in feats if f["sy"] == TEST_SEASON and f["date"] < SPLIT_DATE and f["n_hist"] >= mm.MIN_HIST]
    valid_f = [f for f in feats if f["sy"] == TEST_SEASON and f["date"] >= SPLIT_DATE and f["n_hist"] >= mm.MIN_HIST]
    fit_f = [f for f in feats if f["date"] < SPLIT_DATE and f["n_hist"] >= mm.MIN_HIST]   # minutes model uses 2024-25 too
    model, lam = minutes_setup(fit_f, names)
    fit_pred = mm.predict(model, mm.design(fit_f, names))
    qtab = mm.residual_quantiles(fit_pred, np.array([f["y"] for f in fit_f]))
    lookup = {(r["date"], r["aid"]): r for r in rows}
    tr = attach_actuals(make_items(train_f, mm.predict(model, mm.design(train_f, names)), qtab, rmap), lookup)
    va = attach_actuals(make_items(valid_f, mm.predict(model, mm.design(valid_f, names)), qtab, rmap), lookup)
    log(f"Simulation backtest: fit on {len(tr['mean'])} player-games before {dmy(SPLIT_DATE)}, "
        f"check on {len(va['mean'])} player-games from {dmy(SPLIT_DATE)} (2025-26)")
    if len(tr["mean"]) < MIN_FIT_ROWS or len(va["mean"]) < 500:
        raise SystemExit("not enough rows for the simulation backtest")

    P = fit_params(tr)
    log(f"  fitted: points exponent {P['delta'][0]} (kept={P['notes']['pts_coupling']['kept']}), "
        f"rebounds exponent {P['delta'][1]} (kept={P['notes']['reb_coupling']['kept']}), "
        f"assists exponent {P['delta'][2]} (kept={P['notes']['ast_coupling']['kept']}); "
        f"phi pts/reb/ast {[float(x) for x in P['phi']]}; zero-points share {P['pi_pts']}")

    log("  simulating the check period ...")
    full = evaluate(va, P, N_BACKTEST, SEED)
    y3 = va["y"]
    ypts, ypra = y3[:, 0], y3.sum(1)
    report = {"fit_rows": len(tr["mean"]), "check_rows": len(va["mean"]), "draws_per_game": N_BACKTEST,
              "fit_period": f"2025-26 before {dmy(SPLIT_DATE)}", "check_period": f"2025-26 from {dmy(SPLIT_DATE)}",
              "settings": {k: P[k] for k in ("delta", "c", "phi", "pi_pts")}, "settings_notes": P["notes"]}

    # --- accuracy vs simple baselines
    base = baseline_table(rows, zip(va["date"], va["aid"]))
    keys = list(zip(va["date"], va["aid"]))
    b_l10 = np.array([np.mean(base[k][0][-10:], axis=0) if base[k][0] else (np.nan,) * 3 for k in keys], dtype=float)
    b_szn = np.array([base[k][1] if base[k][1] else (np.nan,) * 3 for k in keys], dtype=float)
    ok = ~(np.isnan(b_l10).any(1) | np.isnan(b_szn).any(1))
    report["accuracy"] = {"rows_compared": int(ok.sum())}
    for s, y, sel in (("pts", ypts, 0), ("pra", ypra, None)):
        l10 = b_l10[:, 0] if sel == 0 else b_l10.sum(1)
        sz = b_szn[:, 0] if sel == 0 else b_szn.sum(1)
        r = full[s]
        acc = {"simulated_median_MAE": round(float(np.mean(np.abs(r["med"][ok] - y[ok]))), 3),
               "last10_average_MAE": round(float(np.mean(np.abs(l10[ok] - y[ok]))), 3),
               "season_average_MAE": round(float(np.mean(np.abs(sz[ok] - y[ok]))), 3),
               "simulated_mean_RMSE": round(float(np.sqrt(np.mean((r["mean"][ok] - y[ok]) ** 2))), 3),
               "last10_average_RMSE": round(float(np.sqrt(np.mean((l10[ok] - y[ok]) ** 2))), 3),
               "season_average_RMSE": round(float(np.sqrt(np.mean((sz[ok] - y[ok]) ** 2))), 3)}
        best = min(l10, sz, key=lambda b: np.mean((b[ok] - y[ok]) ** 2))
        d, z = paired((r["mean"][ok] - y[ok]) ** 2, (best[ok] - y[ok]) ** 2)
        acc["mean_vs_best_baseline_squared_error_z"] = round(z, 1)
        acc["median_vs_best_baseline_abs_error_z"] = round(paired(np.abs(r["med"][ok] - y[ok]), np.abs(best[ok] - y[ok]))[1], 1)
        report["accuracy"][s] = acc

    # --- calibration
    report["calibration"] = {}
    for s in ("pts", "pra"):
        r = full[s]
        rel, ece = reliability(r["lp"], r["lo"])
        report["calibration"][s] = {"interval_coverage": coverage(r["pit"]),
                                    "p_over_at_test_lines": rel, "p_over_calibration_error": ece,
                                    "test_lines": "the half-point line nearest the simulated median, and lines 0.35 spread-widths above and below"}
    cov_bins = {}
    for lo, hi in ((0, 12), (12, 20), (20, 28), (28, 60)):
        mk = (va["mean"] >= lo) & (va["mean"] < hi)
        if mk.sum() >= 100:
            cov_bins[f"expected {lo}-{hi} min"] = {"rows": int(mk.sum()),
                                                  "points_p10_p90": round(float(np.mean((full["pts"]["pit"][mk] > .1) & (full["pts"]["pit"][mk] < .9))), 3),
                                                  "pra_p10_p90": round(float(np.mean((full["pra"]["pit"][mk] > .1) & (full["pra"]["pit"][mk] < .9))), 3)}
    report["calibration"]["by_expected_minutes"] = cov_bins

    # --- P(over) against a naive frequency baseline (share of last 20 games over the line)
    report["vs_frequency_baseline"] = {}
    for s, col in (("pts", 0), ("pra", 3)):
        r = full[s]
        have = np.array([len(base[k][0]) >= 10 for k in zip(va["date"], va["aid"])])
        pb = np.zeros_like(r["lp"])
        for i, k in enumerate(zip(va["date"], va["aid"])):
            if not have[i]:
                continue
            vals = np.array([h[0] if s == "pts" else sum(h) for h in base[k][0]], dtype=float)
            for q in range(3):
                pb[i, q] = (np.sum(vals > r["lines"][i, q]) + 1.0) / (len(vals) + 2.0)
        bm, bb = brier_rows(r["lp"][have], r["lo"][have]), brier_rows(pb[have], r["lo"][have])
        d, z = paired(bm, bb)
        report["vs_frequency_baseline"][s] = {"rows": int(have.sum()), "brier_simulation": round(float(bm.mean()), 4),
                                              "brier_last20_frequency": round(float(bb.mean()), 4), "z": round(z, 1),
                                              "note": "lower is better; z < -2 means the simulation is clearly better"}

    # --- do the pieces earn their keep? remove one at a time, same random numbers
    log("  ablations (removing one piece at a time) ...")
    ablations = {}
    variants = {
        "without minutes coupling": (fit_params(tr, no_coupling=True), False),
        "without zero-points share": (fit_params(tr, no_zero=True), False),
        "minutes fixed at their mean (no minutes spread)": (P, True),
        "Poisson counts (no overdispersion)": (dict(P, phi=[1.0, 1.0, 1.0], pi_pts=0.0), False),
    }
    for name, (Pv, pm) in variants.items():
        alt = evaluate(va, Pv, N_BACKTEST, SEED, point_minutes=pm,
                       fixed_lines={s_: full[s_]["lines"] for s_ in ("pts", "pra")})
        row = {}
        for s in ("pts", "pra"):
            dc, zc = paired(alt[s]["crps"], full[s]["crps"])
            db, zb = paired(brier_rows(alt[s]["lp"], alt[s]["lo"]), brier_rows(full[s]["lp"], full[s]["lo"]))
            row[s] = {"crps_change": round(dc, 4), "crps_z": round(zc, 1), "brier_change": round(db, 4), "brier_z": round(zb, 1),
                      "p10_p90_coverage": coverage(alt[s]["pit"])["p10_p90"]}
        ablations[name] = row
    report["ablations_vs_final"] = ablations
    report["ablations_note"] = ("positive change = worse than the final model; all variants answer the same lines. Tested earlier and not kept: extra rate-uncertainty "
                                "draws (no change), a correlation copula between points/rebounds/assists (see below), and forcing team "
                                "minutes to 240 (see below).")

    # --- leftover correlation between stats, given minutes (why no copula)
    rho = np.maximum(va["m"] / np.maximum(va["mean"], 1.0), RHO_FLOOR)
    mu = np.stack([P["c"][s] * va["rates"][:, s] * va["m"] * rho ** P["delta"][s] for s in range(3)], axis=1)
    z = (y3 - mu) / np.sqrt(np.array(P["phi"])[None, :] * np.maximum(mu, 0.05))
    cm = np.corrcoef(z.T)
    ratio = float(np.sum((y3.sum(1) - mu.sum(1)) ** 2) / np.sum(np.array(P["phi"])[None, :] * mu))
    report["stat_correlation_given_minutes"] = {
        "points_rebounds": round(float(cm[0, 1]), 3), "points_assists": round(float(cm[0, 2]), 3),
        "rebounds_assists": round(float(cm[1, 2]), 3),
        "pra_variance_vs_independent_sum": round(ratio, 3),
        "note": "ratio near 1 = no extra correlation to model; shared minutes already correlate the three stats"}

    # --- team minutes check (why no forced 240)
    tot_p, tot_a = defaultdict(float), defaultdict(float)
    for d, t, p, mn in zip(va["date"], va["team"], va["mean"], va["m"]):
        tot_p[(d, t)] += p
        tot_a[(d, t)] += mn
    diff = np.array([tot_p[k] - tot_a[k] for k in tot_p])
    report["team_minutes_check"] = {
        "team_games": len(diff), "mean_predicted_total_for_players_who_played": round(float(np.mean(list(tot_p.values()))), 1),
        "mean_actual_total_same_players": round(float(np.mean(list(tot_a.values()))), 1),
        "mean_difference": round(float(diff.mean()), 2),
        "note": "average bias is about half a percent, so a forced 240 adds little; it also needs complete rosters, which early in a season are not available"}

    report["seconds"] = round(time.time() - t0)
    log("\nResults (check period):")
    for s in ("pts", "pra"):
        a, c, v = report["accuracy"][s], report["calibration"][s], report["vs_frequency_baseline"][s]
        log(f"  {s.upper():<4} median MAE {a['simulated_median_MAE']} vs last-10 avg {a['last10_average_MAE']} / season avg {a['season_average_MAE']} | "
            f"mean RMSE {a['simulated_mean_RMSE']} vs {a['last10_average_RMSE']} / {a['season_average_RMSE']} (z {a['mean_vs_best_baseline_squared_error_z']})")
        log(f"       80% interval holds {c['interval_coverage']['p10_p90']}, 50% interval holds {c['interval_coverage']['p25_p75']}; "
            f"P(over) calibration error {c['p_over_calibration_error']}; Brier {v['brier_simulation']} vs last-20 frequency {v['brier_last20_frequency']} (z {v['z']})")
    log("  Pieces (change when removed; positive = worse without it):")
    for k, v in ablations.items():
        log(f"    {k}")
        for s_ in ("pts", "pra"):
            x = v[s_]
            log(f"      {s_.upper():<4} CRPS {x['crps_change']:+.4f} (z {x['crps_z']}), P(over) Brier {x['brier_change']:+.4f} (z {x['brier_z']}), "
                f"80% interval holds {x['p10_p90_coverage']}")
    log(f"  leftover stat correlation given minutes: {report['stat_correlation_given_minutes']}")
    log(f"  team minutes: {report['team_minutes_check']['mean_predicted_total_for_players_who_played']} predicted vs "
        f"{report['team_minutes_check']['mean_actual_total_same_players']} actual (same players)")
    DATA.mkdir(exist_ok=True)
    SIM_BACKTEST.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"Wrote {SIM_BACKTEST} ({report['seconds']} s)")
    return report


# ---------- projection ----------

def read_projection():
    if not MIN_PROJECTION.exists():
        raise SystemExit(f"{MIN_PROJECTION} is missing: run the daily update (or Minutes backtest and re-fit) first")
    with open(MIN_PROJECTION, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        missing = [c for c in REQUIRED_COLS if c not in cols]
        if missing:
            raise SystemExit(f"{MIN_PROJECTION} has the old layout (missing {missing}). Run the daily update so "
                             f"minutes_model.py rewrites it, then run this again")
        return list(rd)


def read_rates():
    if not PLAYER_RATES.exists():
        raise SystemExit(f"{PLAYER_RATES} is missing: run baseline_rates.py --mode estimate first")
    out = {}
    with open(PLAYER_RATES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["athlete_id"]] = r
    return out


def cdf_table(counts_row, n):
    """P(X <= k) for k = 0, 1, 2 ... until it reaches ~1."""
    cdf = np.cumsum(np.bincount(counts_row)) / n
    stop = int(np.argmax(cdf >= CDF_DONE)) + 1
    return [round(float(v), 4) for v in cdf[:stop]]


def run_project(rows):
    t0 = time.time()
    errors, warnings = [], []
    today_et = datetime.now(timezone.utc).astimezone(_et()).date()
    proj = read_projection()
    rates_by_id = read_rates()
    rp, _ = br.load_params()
    names = json.loads(mm.PARAMS.read_text(encoding="utf-8"))["features"] if mm.PARAMS.exists() else list(mm.FEATURES)

    feats, _ = mm.walk(rows)
    usable = [f for f in feats if f["n_hist"] >= mm.MIN_HIST]
    model, lam = minutes_setup(usable, names)
    pred_all = mm.predict(model, mm.design(usable, names))
    qtab = mm.residual_quantiles(pred_all, np.array([f["y"] for f in usable]))
    seasons = {r["sy"] for r in rows if r["sy"] >= TEST_SEASON}
    rmap = rate_snapshots(rows, rp, seasons)
    lookup = {(r["date"], r["aid"]): r for r in rows}
    it = attach_actuals(make_items(usable, pred_all, qtab, rmap), lookup)
    if len(it["mean"]) < MIN_FIT_ROWS:
        raise SystemExit(f"only {len(it['mean'])} past player-games to fit the simulation settings (need {MIN_FIT_ROWS})")
    P = fit_params(it)
    last_game = max(r["date"] for r in rows)
    SIM_PARAMS.write_text(json.dumps({
        "settings": {k: P[k] for k in ("delta", "c", "phi", "pi_pts")}, "notes": P["notes"], "fit_rows": P["fit_rows"],
        "fitted_through": dmy(last_game), "fitted_at": dmy_time(datetime.now(timezone.utc)) + " UTC"}, indent=2), encoding="utf-8")
    log(f"Fitted simulation settings on {P['fit_rows']} past player-games through {dmy(last_game)}: "
        f"points exponent {P['delta'][0]}, phi {P['phi']}, zero-points share {P['pi_pts']}")

    # freshness of the inputs
    rate_dates = [r["last_game_date"] for r in rates_by_id.values() if r.get("last_game_date")]
    if rate_dates and max(rate_dates) < last_game:
        warnings.append(f"player_rates.csv is older than the box scores (rates to {dmy(max(rate_dates))}, games to {dmy(last_game)})")

    games_ok = [r for r in proj if datetime.strptime(r["game_date_et"], "%d/%m/%Y").date() >= today_et]
    stale = len(proj) - len(games_ok)
    if not proj:
        warnings.append("minutes_projection.csv has no rows")
    elif not games_ok:
        warnings.append(f"every row in minutes_projection.csv is for a game before {today_et.strftime('%d/%m/%Y')}: no upcoming games, or the minutes step did not refresh")
    elif stale:
        warnings.append(f"{stale} rows for games already past were skipped")
    proj = games_ok

    # league-average rates as a fallback for players the rates file does not know
    tot_m = sum(float(r["min_last_season"] or 0) + float(r["min_this_season"] or 0) for r in rates_by_id.values()) or 1.0
    league = [sum(float(r[f"{s}_per_min"] or 0) * (float(r["min_last_season"] or 0) + float(r["min_this_season"] or 0))
                  for r in rates_by_id.values()) / tot_m for s in STATS]

    rng = np.random.default_rng(SEED)
    out_rows, players, no_rates = [], [], 0
    bs = 50
    for a in range(0, len(proj), bs):
        chunk = proj[a:a + bs]
        q7 = np.array([[float(r[k]) for k in ("p05", "p10", "p25", "p50", "p75", "p90", "p95")] for r in chunk])
        mean = np.array([float(r["proj_min"]) for r in chunk])
        rts, fallback = [], []
        for r in chunk:
            rr = rates_by_id.get(r["athlete_id"])
            if rr is None:
                rts.append(league)
                fallback.append(1)
            else:
                rts.append([float(rr[f"{s}_per_min"]) for s in STATS])
                fallback.append(0)
        d = simulate(q7, mean, np.array(rts), P, N_PROJECT, rng)
        for i, r in enumerate(chunk):
            rec = {"game_date_et": r["game_date_et"], "event_id": r["event_id"], "team": r["team"], "opp": r["opp"], "home": int(r["home"]),
                   "athlete_id": r["athlete_id"], "player": r["player"], "status": r["status"], "status_source": r["status_source"],
                   "p_play": float(r["p_play"]), "proj_min": float(r["proj_min"]),
                   "min_p10": float(r["p10"]), "min_p90": float(r["p90"])}
            js = {"id": r["athlete_id"], "name": r["player"], "team": r["team"], "opp": r["opp"], "home": int(r["home"]),
                  "game_date": r["game_date_et"], "event_id": r["event_id"], "status": r["status"], "status_source": r["status_source"],
                  "p_play": float(r["p_play"]), "minutes": {"mean": float(r["proj_min"]), "p10": float(r["p10"]), "p90": float(r["p90"])}}
            for s in ("pts", "pra"):
                x = d[s][i]
                qs = np.quantile(x, [0.10, 0.25, 0.50, 0.75, 0.90])
                rec.update({f"{s}_mean": round(float(x.mean()), 2), f"{s}_p10": float(qs[0]), f"{s}_p25": float(qs[1]),
                            f"{s}_p50": float(qs[2]), f"{s}_p75": float(qs[3]), f"{s}_p90": float(qs[4])})
                js[s] = {"mean": round(float(x.mean()), 2), "p10": float(qs[0]), "p25": float(qs[1]), "p50": float(qs[2]),
                         "p75": float(qs[3]), "p90": float(qs[4]), "cdf": cdf_table(x, N_PROJECT)}
            for s in ("reb", "ast"):
                rec[f"{s}_mean"] = round(float(d[s][i].mean()), 2)
                js[f"{s}_mean"] = rec[f"{s}_mean"]
            rec["rates_fallback"] = fallback[i]
            rec["low_history"] = int(r.get("low_history") or 0)
            no_rates += fallback[i]
            out_rows.append(rec)
            js["flags"] = {"low_history": rec["low_history"], "league_average_rates_used": fallback[i]}
            players.append(js)

    team_tot = defaultdict(float)
    for r in out_rows:
        team_tot[(r["game_date_et"], r["event_id"], r["team"])] += r["p_play"] * r["proj_min"]
    tt = list(team_tot.values())
    if no_rates:
        warnings.append(f"{no_rates} player-games used league-average rates (player not in player_rates.csv)")
    now = datetime.now(timezone.utc)
    DATA.mkdir(exist_ok=True)
    if out_rows:
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
    else:
        OUT_CSV.write_text("", encoding="utf-8")
    OUT_JSON.write_text(json.dumps({
        "generated": dmy_time(now) + " UTC", "generated_epoch": int(now.timestamp()),
        "data_through": dmy(last_game), "draws_per_player_game": N_PROJECT,
        "note": "All stat numbers assume the player plays (books normally void the prop otherwise); p_play is the chance he plays. "
                "cdf[k] = P(stat <= k) for k = 0, 1, 2 ...; P(over a x.5 line) = 1 - cdf[floor(line)].",
        "players": players}, separators=(",", ":")), encoding="utf-8")
    nonfinite = [r["player"] for r in out_rows if not all(math.isfinite(float(r[k])) for k in ("pts_mean", "pra_mean"))]
    if nonfinite:
        errors.append(f"non-finite results for {nonfinite[:5]}")
    games = len({(r["game_date_et"], r["event_id"]) for r in out_rows})
    health = {
        "run": dmy_time(now) + " UTC", "ok": not errors, "errors": errors, "error_count": len(errors),
        "warnings": warnings, "warning_count": len(warnings),
        "player_games": len(out_rows), "games": games, "game_dates": sorted({r["game_date_et"] for r in out_rows},
                                                                           key=lambda s: datetime.strptime(s, "%d/%m/%Y")),
        "data_through": dmy(last_game), "fit_rows": P["fit_rows"], "seconds": round(time.time() - t0),
        "team_expected_minutes_total": ({"mean": round(float(np.mean(tt)), 1), "min": round(float(np.min(tt)), 1),
                                         "max": round(float(np.max(tt)), 1)} if tt else None),
    }
    HEALTH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    log(f"Projected {len(out_rows)} player-games across {games} games ({N_PROJECT} draws each). Wrote {OUT_CSV}, {OUT_JSON}")
    for w_ in warnings:
        log(f"  WARNING: {w_}")
    for e_ in errors:
        log(f"  ERROR: {e_}")
    return health


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["backtest", "project", "both"], required=True)
    args = p.parse_args(argv)
    rows = load_rows()
    log(f"Loaded {len(rows)} usable player-games, latest game {dmy(rows[-1]['date'])}")
    health = None
    if args.mode in ("backtest", "both"):
        run_backtest(rows)
    if args.mode in ("project", "both"):
        try:
            health = run_project(rows)
        except SystemExit as e:
            msg = str(e)
            DATA.mkdir(exist_ok=True)
            HEALTH.write_text(json.dumps({"run": dmy_time(datetime.now(timezone.utc)) + " UTC", "ok": False, "errors": [msg],
                                          "error_count": 1, "warnings": [], "warning_count": 0}, indent=2), encoding="utf-8")
            raise
    return 0 if (health is None or health["ok"]) else 1


if __name__ == "__main__":
    sys.exit(main())
