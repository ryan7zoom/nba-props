"""Tests for the parts that decide money: probabilities, prices and the calibration map.

Run from the repository root:

    python -m pytest tests -q

These do not touch the network or the data/ folder; they build small synthetic cases so a change in the maths shows
up before a change in a bankroll does.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import alt_lines          # noqa: E402
import calibrate as cal   # noqa: E402
import grade_lines as gl  # noqa: E402
import shortlist          # noqa: E402
import team_totals        # noqa: E402


# ------------------------------------------------------------------ prices

def test_decimal_from_american():
    assert gl.dec(-100) == pytest.approx(2.0)
    assert gl.dec(100) == pytest.approx(2.0)
    assert gl.dec(-110) == pytest.approx(1.909, abs=1e-3)
    assert gl.dec(150) == pytest.approx(2.5)


def test_devig_proportional_is_symmetric():
    assert gl.fair(-110, -110) == pytest.approx(0.5)
    assert gl.fair(150, -170) > 0.35


def test_devig_power_sums_to_one_and_is_symmetric():
    for a, b in ((-110, -110), (150, -170), (-250, 205), (100, -120)):
        p = gl.fair_power(a, b)
        assert 0.0 < p < 1.0
        assert p + gl.fair_power(b, a) == pytest.approx(1.0, abs=1e-9)
    assert gl.fair_power(-110, -110) == pytest.approx(0.5, abs=1e-9)


# ------------------------------------------------------------------ side probabilities

CDF = [0.05, 0.15, 0.40, 0.70, 0.90, 0.97, 0.995, 1.0]


def test_half_point_lines_never_push():
    assert shortlist.p_side(CDF, 2.5, "u") == pytest.approx(0.40)      # P(X <= 2)
    assert shortlist.p_side(CDF, 2.5, "o") == pytest.approx(0.60)
    assert shortlist.p_side(CDF, 2.5, "u") + shortlist.p_side(CDF, 2.5, "o") == pytest.approx(1.0)


def test_integer_lines_are_push_adjusted():
    # at line 3: push = P(X = 3) = 0.30, so the two sides are 0.40/0.70 and 0.30/0.70
    under = shortlist.p_side(CDF, 3, "u")
    over = shortlist.p_side(CDF, 3, "o")
    assert under == pytest.approx(0.40 / 0.70)
    assert over == pytest.approx(0.30 / 0.70)
    assert under + over == pytest.approx(1.0)


def test_sides_are_monotone_in_the_line():
    prev_u = -1.0
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        u = shortlist.p_side(CDF, line, "u")
        assert u >= prev_u - 1e-12          # an Under gets safer as the line rises
        prev_u = u


# ------------------------------------------------------------------ the safety ladder

def test_tightest_under_is_the_lowest_line_that_still_clears_the_level():
    line, p = alt_lines.tightest_under(CDF, 0.90)
    assert line == 4.5 and p == pytest.approx(0.90)         # cdf[4] = 0.90
    assert alt_lines.tightest_under(CDF[:2], 0.99) is None


def test_tightest_over_is_the_highest_line_that_still_clears_the_level():
    line, p = alt_lines.tightest_over(CDF, 0.90)
    best = None                                  # the highest j with 1 - cdf[j] >= 0.90
    for j in range(len(CDF)):
        if 1 - CDF[j] >= 0.90:
            best = (j + 0.5, 1 - CDF[j])
    assert (line, p) == pytest.approx(best)
    assert line == 0.5 and p == pytest.approx(0.95)


def test_min_price_covers_the_edge():
    p = 0.90
    m = alt_lines.min_price(p, 0.02)
    assert m == pytest.approx(1.02 / 0.90, abs=1e-3)      # rounded to 3 decimals
    assert p * m - 1 == pytest.approx(0.02, abs=2e-3)      # rounded to 3 decimals


def test_ladder_probabilities_complement_and_are_monotone():
    rows = team_totals.ladder(mu=230.0, sigma=18.0, main=228.5)
    assert rows
    for r in rows:
        assert r["p_over"] + r["p_under"] == pytest.approx(1.0, abs=1e-9)
    overs = [r["p_over"] for r in rows]
    assert all(a >= b - 1e-12 for a, b in zip(overs, overs[1:]))   # the Over gets harder as the line rises


# ------------------------------------------------------------------ the calibration map

def test_map_without_knots_is_the_identity():
    for p in (0.01, 0.25, 0.5, 0.83, 0.999):
        assert cal.apply_map(p, []) == pytest.approx(p)


def test_map_endpoints_and_monotonicity():
    knots = [[0.0, 0.0], [0.36, 0.46], [0.56, 0.48], [1.0, 1.0]]
    assert cal.apply_map(0.0, knots) == pytest.approx(0.0)
    assert cal.apply_map(1.0, knots) == pytest.approx(1.0)
    grid = [i / 200 for i in range(201)]
    vals = [cal.apply_map(p, knots) for p in grid]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    # the over-confident middle is pulled down, and a 90% line stays close to 90%
    assert cal.apply_map(0.56, knots) == pytest.approx(0.48, abs=1e-9)
    assert 0.85 <= cal.apply_map(0.95, knots) <= 0.96


def test_fit_map_recovers_a_known_shrink():
    # our model says p, reality happens at a compressed version of it
    rows = []
    n = 100
    for p in [i / 100 for i in range(20, 80)]:
        q = 0.5 + 0.5 * (p - 0.5)               # the truth is the model squashed halfway to 50%
        wins = int(round(q * n))
        for i in range(n):
            rows.append({"ours_o": p, "over_won": i < wins})
    knots, detail = cal.fit_map(rows)
    assert knots and detail
    assert cal.apply_map(0.30, knots) > 0.30      # low numbers were understated
    assert cal.apply_map(0.70, knots) < 0.70      # high numbers were overstated


def test_apply_calibration_stays_in_range():
    calib = {"by_market": {"pts": {"knots": [[0.0, 0.0], [0.5, 0.45], [1.0, 1.0]], "w": 0.4}}}
    for p in (0.001, 0.4, 0.99, 1.0):
        for fair in (None, 0.001, 0.5, 0.999):
            out = cal.apply_calibration(p, "pts", calib, fair)
            assert 0.005 <= out <= 0.995
    # no market price -> the correction stands alone
    only_map = cal.apply_calibration(0.5, "pts", calib, None)
    assert only_map == pytest.approx(0.45, abs=1e-9)
    # with a price -> blended toward it
    assert cal.apply_calibration(0.5, "pts", calib, 0.6) == pytest.approx(0.4 * 0.45 + 0.6 * 0.6)


def test_unknown_market_falls_back_to_the_global_spec():
    calib = {"global": {"knots": [], "w": 0.5}, "by_market": {}}
    assert cal.apply_calibration(0.5, "reb", calib, 0.5) == pytest.approx(0.5)


# ------------------------------------------------------------------ appearance priors

def test_appear_prior_lookup():
    table = {"buckets": [{"min_recent_minutes": 20.0, "appearance_rate": 0.88},
                         {"min_recent_minutes": 0.0, "appearance_rate": 0.42}]}
    assert cal.appear_prior(30, table) == pytest.approx(0.88)
    assert cal.appear_prior(20, table) == pytest.approx(0.88)
    assert cal.appear_prior(5, table) == pytest.approx(0.42)


def test_bottom_up_is_points_per_240_minutes():
    proj = {"players": [
        {"event_id": "e", "team": "BOS", "p_play": 1.0, "p_appear": 1.0,
         "pts": {"mean": 24.0}, "minutes": {"mean": 24.0}},
        {"event_id": "e", "team": "BOS", "p_play": 1.0, "p_appear": 0.5,
         "pts": {"mean": 12.0}, "minutes": {"mean": 24.0}},
    ]}
    per240, mins, n = team_totals.bottom_up(proj)
    # points per expected minute: (24 + 0.5*12) / (24 + 0.5*24) = 30/36
    assert per240[("e", "BOS")] == pytest.approx(240 * 30 / 36)
    assert mins[("e", "BOS")] == pytest.approx(36.0)
