"""Opponent defense: how much easier or harder each team makes it to score, rebound and assist.

Idea: for every past game, compare what a player actually produced with what his own per-minute
rate says he should have produced in the minutes he played. Add that gap up per OPPONENT. A team
whose opponents keep falling short of their own rates is a good defense; a team they beat is a
soft one. Because the gap is measured against each player's own rate, a team that only met
stars is not unfairly marked as bad.

The number for a team is a percentage (for example +9% points). It is
  - shrunk toward zero when a team has few games (K, measured in player-games),
  - weighted toward recent games (half-life H team games),
  - blended with last season at weight a, which carries early-season predictions.
Then the simulation multiplies each player's per-minute rates by (1 + effect) for his opponent.

Modes
  backtest  pick K, H, a on 2025-26 before 15/01/2026, then check on the rest of 2025-26
            (each game only sees games before it). Keeps defense only if it clearly helps.
            Writes data/defense_params.json and data/defense_backtest.json.
  estimate  compute today's effects for every team. Writes data/defense_effects.csv.
  both      backtest, then estimate.

Needs numpy.
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import baseline_rates as br
import minutes_model as mm
import simulate_props as sp
from injury_report import dmy, dmy_time

DATA = Path("data")
EFFECTS = DATA / "defense_effects.csv"
PARAMS = DATA / "defense_params.json"
BACKTEST = DATA / "defense_backtest.json"

TEAMS = sorted(mm.NBA)
TIDX = {t: i for i, t in enumerate(TEAMS)}
STATS = ("pts", "reb", "ast")
K_GRID = [10, 25, 50, 100, 200, 400, 800]
H_GRID = [5, 10, 20, 40, 1e9]
A_GRID = [0.0, 0.25, 0.5, 1.0]
DEFAULTS = {"K": 400, "H": 40, "a": 0.5}
CLIP = 0.30
KEEP_Z = 2.0                 # defense kept only if it improves the check period by this many standard errors
EARLY_DAYS = 30
CRUDE_K_MIN = 100.0          # prior strength (minutes) for the simple rates used for 2024-25
CRUDE_MIN_HISTORY = 100.0    # minutes a player needs before his 2024-25 games count


def log(msg=""):
    print(msg, flush=True)


# ---------- simple per-minute rates for seasons the step 2 model does not cover (2024-25) ----------

def crude_rates(rows, seasons):
    """Player's own season-to-date per-minute rates, shrunk toward the league, as known before each game."""
    out = {}
    pm = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])      # (aid, season) -> minutes, pts, reb, ast
    lg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])      # season -> same, league-wide
    i = 0
    while i < len(rows):
        j = i
        while j < len(rows) and rows[j]["date"] == rows[i]["date"]:
            j += 1
        for r in rows[i:j]:
            if r["sy"] not in seasons:
                continue
            a, L = pm[(r["aid"], r["sy"])], lg[r["sy"]]
            if a[0] >= CRUDE_MIN_HISTORY and L[0] > 0:
                out[(r["date"], r["aid"])] = tuple((CRUDE_K_MIN * L[s + 1] / L[0] + a[s + 1]) / (CRUDE_K_MIN + a[0]) for s in range(3))
        for r in rows[i:j]:
            if r["sy"] not in seasons:
                continue
            for acc in (pm[(r["aid"], r["sy"])], lg[r["sy"]]):
                acc[0] += r["min"]
                for s in range(3):
                    acc[s + 1] += r["y"][s]
        i = j
    return out


def make_records(rows, rate_of, seasons):
    keep = []
    for r in rows:
        if r["sy"] in seasons:
            rt = rate_of.get((r["date"], r["aid"]))
            if rt is not None:
                keep.append((r, rt))
    rate = np.array([rt for r, rt in keep], dtype=float).reshape(-1, 3)
    m = np.array([r["min"] for r, rt in keep], dtype=float)
    y = np.array([r["y"] for r, rt in keep], dtype=float).reshape(-1, 3)
    exp = rate * m[:, None]
    return {
        "date": [r["date"] for r, rt in keep], "aid": [r["aid"] for r, rt in keep],
        "day": np.array([datetime.fromisoformat(r["date"]).toordinal() for r, rt in keep]),
        "sy": np.array([r["sy"] for r, rt in keep]),
        "opp": np.array([TIDX[r["opp"]] for r, rt in keep]),
        "rate": rate, "m": m, "y": y, "exp": exp, "resid": y - exp,
    }


# ---------- the walk: each game sees only games before it ----------

def walk_effects(rec, K, H, a, ebar):
    """Effect (fraction) of each record's opponent on points, rebounds, assists, using earlier games only.
    Also returns the effect for every team as of the end, and each team's game count this season."""
    n = len(rec["day"])
    eff = np.zeros((n, 3))
    cur_r, cur_e = np.zeros((30, 3)), np.zeros((30, 3))
    tot_r, tot_e = np.zeros((30, 3)), np.zeros((30, 3))
    prv_r, prv_e = np.zeros((30, 3)), np.zeros((30, 3))
    sy_t = np.zeros(30, dtype=int)
    games = np.zeros(30, dtype=int)
    delta = 0.5 ** (1.0 / H) if H < 1e8 else 1.0
    day = rec["day"]
    starts = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    kb = K * ebar
    for b in range(len(starts) - 1):
        lo, hi = starts[b], starts[b + 1]
        opps = np.unique(rec["opp"][lo:hi])
        sy_now = rec["sy"][lo]
        for t in opps:
            if sy_t[t] != sy_now:
                if sy_t[t] != 0:
                    prv_r[t], prv_e[t] = tot_r[t].copy(), tot_e[t].copy()
                cur_r[t], cur_e[t], tot_r[t], tot_e[t] = 0.0, 0.0, 0.0, 0.0
                games[t] = 0
                sy_t[t] = sy_now
        e_all = (cur_r + a * prv_r) / np.maximum(cur_e + a * prv_e + kb, 1e-9)
        eff[lo:hi] = e_all[rec["opp"][lo:hi]]
        add_r, add_e = np.zeros((30, 3)), np.zeros((30, 3))
        np.add.at(add_r, rec["opp"][lo:hi], rec["resid"][lo:hi])
        np.add.at(add_e, rec["opp"][lo:hi], rec["exp"][lo:hi])
        for t in opps:
            cur_r[t] = cur_r[t] * delta + add_r[t]
            cur_e[t] = cur_e[t] * delta + add_e[t]
            tot_r[t] += add_r[t]
            tot_e[t] += add_e[t]
            games[t] += 1
    final = (cur_r + a * prv_r) / np.maximum(cur_e + a * prv_e + kb, 1e-9)
    return eff, final, games


# ---------- scoring ----------

def deviance_rows(rec, eff, mask):
    """Poisson deviance per record and stat, using the player's rate and ACTUAL minutes."""
    mu = np.maximum(rec["exp"][mask] * (1.0 + np.clip(eff[mask], -CLIP, CLIP)), 1e-9)
    y = rec["y"][mask]
    return 2.0 * (np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu), 0.0) - (y - mu))


def normalised_gain(dev_new, dev_old):
    """Per-row improvement (negative = defense helps), each stat scaled by its average deviance."""
    scale = dev_old.mean(0)
    return ((dev_new - dev_old) / scale).sum(1)


def zscore(d):
    se = d.std(ddof=1) / math.sqrt(len(d))
    return float(d.mean() / se) if se > 0 else float("nan")


def window_masks(rec):
    split = datetime.fromisoformat(sp.SPLIT_DATE).toordinal()
    in26 = rec["sy"] == sp.TEST_SEASON
    tune = in26 & (rec["day"] < split)
    check = in26 & (rec["day"] >= split)
    first = rec["day"][in26].min() if in26.any() else 0
    early = in26 & (rec["day"] < first + EARLY_DAYS)
    return tune, check, early


# ---------- backtest ----------

def run_backtest(rows):
    t0 = time.time()
    prep = sp.prepare_backtest(rows)
    rp = prep["rp"]
    rmap = sp.rate_snapshots(rows, rp, [sp.TEST_SEASON])
    prev_season = sp.TEST_SEASON - 1
    crude = crude_rates(rows, {prev_season})
    rate_of = dict(crude)
    rate_of.update(rmap)
    rec = make_records(rows, rate_of, {prev_season, sp.TEST_SEASON})
    ebar = rec["exp"][rec["sy"] == sp.TEST_SEASON].mean(0)
    tune, check, early = window_masks(rec)
    log(f"Defense backtest: {int(tune.sum())} player-games to tune on (before {dmy(sp.SPLIT_DATE)}), "
        f"{int(check.sum())} to check on, last season's games used for the carry-over: {int((rec['sy'] == prev_season).sum())}")
    if tune.sum() < 3000 or check.sum() < 3000:
        raise SystemExit("not enough rows for the defense backtest")

    zero = np.zeros((len(rec["day"]), 3))
    dev0 = {k: deviance_rows(rec, zero, m) for k, m in (("tune", tune), ("check", check), ("early", early))}

    log("  tuning K (shrinkage), H (recent-game half-life), a (last-season weight) ...")
    best = None
    grid = []
    for K in K_GRID:
        for H in H_GRID:
            for a in A_GRID:
                eff, _, _ = walk_effects(rec, K, H, a, ebar)
                g = normalised_gain(deviance_rows(rec, eff, tune), dev0["tune"])
                grid.append((float(g.mean()), K, H, a))
                if best is None or g.mean() < best[0]:
                    best = (float(g.mean()), K, H, a)
    _, K, H, a = best
    eff, final, games = walk_effects(rec, K, H, a, ebar)
    log(f"  tuned: K={K}, half-life={'none' if H > 1e8 else int(H)} games, last-season weight={a}")

    # check period, never used for tuning
    gain_check = normalised_gain(deviance_rows(rec, eff, check), dev0["check"])
    z_check = zscore(gain_check)
    per_stat = {}
    for s, name in enumerate(STATS):
        d = deviance_rows(rec, eff, check)[:, s]
        per_stat[name] = {"deviance_change_pct": round(float(100.0 * (d.mean() / dev0["check"][:, s].mean() - 1.0)), 3),
                          "z": round(float(zscore((d - dev0["check"][:, s]) / dev0["check"][:, s].mean())), 1)}
    # early season: does last season's defense help the first weeks?
    eff_no_prev, _, _ = walk_effects(rec, K, H, 0.0, ebar)
    early_with = normalised_gain(deviance_rows(rec, eff, early), dev0["early"])
    early_without = normalised_gain(deviance_rows(rec, eff_no_prev, early), dev0["early"])
    early_diff = early_with - early_without
    # is the size of the effect right? (slope of the gap on the predicted gap; 1 = right size)
    m = check
    mu0 = rec["exp"][m]
    pred_gap = mu0 * np.clip(eff[m], -CLIP, CLIP)
    slope = [float((pred_gap[:, s] * rec["resid"][m][:, s]).sum() / (pred_gap[:, s] ** 2).sum()) for s in range(3)]

    kept = bool(gain_check.mean() < 0 and z_check <= -KEEP_Z)
    log(f"  check period (from {dmy(sp.SPLIT_DATE)}): fit change {100 * gain_check.mean() / 3:.3f}% per stat on average, "
        f"z {z_check:.1f}  ->  defense {'KEPT' if kept else 'NOT kept'}")

    # does it help the full simulation (the thing that produces the lines)?
    log("  full simulation with and without defense ...")
    tr, va = prep["tr"], prep["va"]
    key = {(d, a_): i for i, (d, a_) in enumerate(zip(rec["date"], rec["aid"]))}

    def adjust(it):
        e = np.array([eff[key[k]] if k in key else (0.0, 0.0, 0.0) for k in zip(it["date"], it["aid"])], dtype=float)
        out = dict(it)
        out["rates"] = it["rates"] * (1.0 + np.clip(e, -CLIP, CLIP))
        return out

    tr_d, va_d = adjust(tr), adjust(va)
    P0, P1 = sp.fit_params(tr), sp.fit_params(tr_d)
    base = sp.evaluate(va, P0, sp.N_BACKTEST, sp.SEED)
    new = sp.evaluate(va_d, P1, sp.N_BACKTEST, sp.SEED, fixed_lines={s_: base[s_]["lines"] for s_ in ("pts", "pra")})
    y3 = va["y"]
    sim = {}
    for s, y in (("pts", y3[:, 0]), ("pra", y3.sum(1))):
        dc, zc = sp.paired(new[s]["crps"], base[s]["crps"])
        db, zb = sp.paired(sp.brier_rows(new[s]["lp"], new[s]["lo"]), sp.brier_rows(base[s]["lp"], base[s]["lo"]))
        sim[s] = {"median_MAE_without": round(float(np.mean(np.abs(base[s]["med"] - y))), 3),
                  "median_MAE_with": round(float(np.mean(np.abs(new[s]["med"] - y))), 3),
                  "mean_RMSE_without": round(float(np.sqrt(np.mean((base[s]["mean"] - y) ** 2))), 3),
                  "mean_RMSE_with": round(float(np.sqrt(np.mean((new[s]["mean"] - y) ** 2))), 3),
                  "crps_change": round(dc, 4), "crps_z": round(zc, 1), "brier_change": round(db, 4), "brier_z": round(zb, 1),
                  "p10_p90_without": sp.coverage(base[s]["pit"])["p10_p90"], "p10_p90_with": sp.coverage(new[s]["pit"])["p10_p90"],
                  "p_over_calibration_error_with": sp.reliability(new[s]["lp"], new[s]["lo"])[1],
                  "p_over_calibration_error_without": sp.reliability(base[s]["lp"], base[s]["lo"])[1]}
    ranked = sorted(zip(TEAMS, final[:, 0]), key=lambda t: t[1])
    report = {
        "tuned_on": f"2025-26 before {dmy(sp.SPLIT_DATE)}", "checked_on": f"2025-26 from {dmy(sp.SPLIT_DATE)}",
        "settings": {"K": K, "H": None if H > 1e8 else H, "a": a},
        "kept": kept, "check_period": {"fit_change_z": round(z_check, 1), "per_stat": per_stat,
                                       "effect_size_slope_pts_reb_ast": [round(x, 2) for x in slope],
                                       "slope_note": "1.0 means the predicted gap is the right size; below 1 means it is too big"},
        "first_30_days_of_season": {"last_season_adds_improvement_z": round(zscore(early_diff), 1),
                                    "rows": int(early.sum()), "note": "negative z = using last season's defense helped the opening weeks"},
        "simulation_with_vs_without": sim,
        "end_of_2025_26_points_effect_pct": {"toughest": [(t, round(float(100 * v), 1)) for t, v in ranked[:5]],
                                             "softest": [(t, round(float(100 * v), 1)) for t, v in ranked[-5:]]},
        "tuning_grid_best_5": [{"fit_change": round(g, 4), "K": k, "H": h if h < 1e8 else None, "a": aa} for g, k, h, aa in sorted(grid)[:5]],
        "seconds": round(time.time() - t0)}
    DATA.mkdir(exist_ok=True)
    BACKTEST.write_text(json.dumps(report, indent=2), encoding="utf-8")
    PARAMS.write_text(json.dumps({"K": K, "H": H if H < 1e8 else 1e9, "a": a, "kept": kept,
                                  "tuned_on": report["tuned_on"], "checked_on": report["checked_on"],
                                  "check_z": round(z_check, 1)}, indent=2), encoding="utf-8")
    log("\nResults (check period):")
    for s in ("pts", "pra"):
        x = sim[s]
        log(f"  {s.upper():<4} median error {x['median_MAE_without']} -> {x['median_MAE_with']} | mean error {x['mean_RMSE_without']} -> {x['mean_RMSE_with']} | "
            f"CRPS {x['crps_change']:+.4f} (z {x['crps_z']}) | P(over) Brier {x['brier_change']:+.4f} (z {x['brier_z']}) | "
            f"80% interval {x['p10_p90_without']} -> {x['p10_p90_with']}")
    log(f"  per stat fit change: {per_stat}")
    log(f"  effect-size slope (pts/reb/ast): {[round(x, 2) for x in slope]} (1.0 = right size)")
    log(f"  first 30 days: last season's defense helps? z = {report['first_30_days_of_season']['last_season_adds_improvement_z']} (negative = yes)")
    log(f"  end of 2025-26, points: toughest {report['end_of_2025_26_points_effect_pct']['toughest']}, softest {report['end_of_2025_26_points_effect_pct']['softest']}")
    log(f"Wrote {BACKTEST} and {PARAMS} ({report['seconds']} s)")
    return report


# ---------- today's effects ----------

def run_estimate(rows):
    params = dict(DEFAULTS)
    if PARAMS.exists():
        p = json.loads(PARAMS.read_text(encoding="utf-8"))
        params.update({k: p[k] for k in ("K", "H", "a") if k in p})
    else:
        log("  no defense_params.json yet, using default settings")
    rp, _ = br.load_params()
    cur = max(r["sy"] for r in rows)
    seasons = {cur - 1, cur}
    rmap = sp.rate_snapshots(rows, rp, seasons)
    rec = make_records(rows, rmap, seasons)
    if (rec["sy"] == cur).sum() == 0:
        raise SystemExit("no games for the current season with known rates")
    ebar = rec["exp"][rec["sy"] == cur].mean(0)
    _, final, games = walk_effects(rec, params["K"], params["H"], params["a"], ebar)
    last = max(r["date"] for r in rows)
    out = []
    for t in TEAMS:
        i = TIDX[t]
        e = np.clip(final[i], -CLIP, CLIP) * 100.0
        out.append({"team": t, "pts_pct": round(float(e[0]), 2), "reb_pct": round(float(e[1]), 2), "ast_pct": round(float(e[2]), 2),
                    "games_this_season": int(games[i]), "data_through": dmy(last)})
    DATA.mkdir(exist_ok=True)
    with open(EFFECTS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    srt = sorted(out, key=lambda r: r["pts_pct"])
    log(f"Defense effects through {dmy(last)} (K={params['K']}, a={params['a']}): "
        f"toughest vs points {[(r['team'], r['pts_pct']) for r in srt[:3]]}, softest {[(r['team'], r['pts_pct']) for r in srt[-3:]]}. Wrote {EFFECTS}")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["backtest", "estimate", "both"], required=True)
    args = p.parse_args(argv)
    rows = sp.load_rows()
    log(f"Loaded {len(rows)} usable player-games, latest game {dmy(rows[-1]['date'])}")
    if args.mode in ("backtest", "both"):
        run_backtest(rows)
    if args.mode in ("estimate", "both"):
        run_estimate(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
