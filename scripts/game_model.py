"""Game model: team scores, margin, total, win chance, and the bet-builder pool per game.

Modes:  python scripts/game_model.py backtest   (walk-forward test on 2025-26, writes data/team_backtest.json)
        python scripts/game_model.py project    (writes data/games_latest.json for upcoming games)
        python scripts/game_model.py            (both)
Needs: numpy, pandas.
"""
import csv, json, math, os, sys, datetime, time
from math import erf, sqrt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
GMT8 = datetime.timezone(datetime.timedelta(hours=8))

# ---- team model settings (tuned on 2025-26 before 15/01/2026) ----
CARRY, K0, LR, HCA, PCARRY, PLR = 0.6, 12, 0.01, 2.0, 0.5, 0.02
FIT_SEASON, FIT_END = 2026, '2026-01-15'   # ESPN season_year 2026 = 2025-26

# ---- builder settings ----
N_SIM = 20000
PTS_GAP, PRA_GAP = 6, 10          # lowest rung = main line minus this (points 6, PRA 10); the search goes up from there
FLOOR_RA = 3                      # bookie's lowest rung for rebounds and for assists
LEG_LO, LEG_HI = 0.556, 0.833   # every leg is priced 1.20 to 1.80 (chance 83.3% down to 55.6%); inside that window the builder takes the HIGHEST chance
# Your bookie's rung ladders (a leg can only sit on a rung that exists there). Points are irregular at the low end and then go up in 5s;
# P+R, P+A and PRA go up in 5s; rebounds, assists and A+R go up in 2s (the 2-step ladders start at the lowest rung assumed above: 3 for reb/ast, 6 for A+R).
LADDER = {
    'pts': [10, 12, 15, 18, 20, 22, 25, 28, 30, 35] + list(range(40, 101, 5)),
    'pr': list(range(5, 151, 5)), 'pa': list(range(5, 151, 5)), 'pra': list(range(5, 151, 5)),
}
# Chance range a player leg may have, per stat. Coarse ladders (points family, 5-step) get a wider price band than the finer 2-step ones:
PLAYER_HI = 0.80                  # player props start at odds 1.25 (chance 80%); team legs start at 1.20 (LEG_HI)
BAND = {k: (LEG_LO, PLAYER_HI) for k in ('pts', 'pr', 'pa', 'pra')}
RUNG_TIE = 0.02                    # two rungs of the same stat within this of each other (measured from the target) count as tied; the LOWER, safer rung wins
STAT_TIE = 0.0                     # same idea when comparing different stats for one player (0 = strictly the closest to the target wins)
TEAM_TARGETS = (0.82,)             # spread / game-total / team-total pool legs: the line just inside the top of the window
MIN_LEG = LEG_LO                     # a main-line Under joins the pool only when at least this likely
FLOOR = 0.30                       # a third leg is only added while the all-legs chance stays above this (two legs are always built when the pool allows)
MAX_LEGS = 5                       # most legs in one game's builder: 3 player props + up to 2 team totals
MAX_TEAM_LEGS = 2                  # spread / total / team-total legs are one bet said three ways, so keep at most this many
MIN_PLAYER_LEGS = 3               # at least this many player props in every game's builder
PLAYER_BONUS = 0.03                # a player leg wins a near-tie against a team leg (you mostly bet player props)
MIN_MAIN = {'pts': 9.5, 'pra': 18.5}       # SMALLEST main line (with a DK/FD line) that can be a builder leg
MIN_MAIN_EST = {'pts': 9.5, 'pra': 24.5, 'reb': 3.5, 'ast': 3.5, 'pr': 12.5, 'pa': 12.5, 'ar': 6.5}   # same, for stats with no DK/FD line (our own median is the anchor, flagged as an estimate)
BOOK = {'draftkings': 'DK', 'fanduel': 'FD'}
STATS = ('pts', 'pr', 'pa', 'pra')   # player legs: only stats that include points (no rebounds / assists / A+R: too streaky); the rung closest to the target price wins
STAT_NAME = {'pts': 'pts', 'reb': 'reb', 'ast': 'ast', 'pr': 'P+R', 'pa': 'P+A', 'ar': 'A+R', 'pra': 'PRA'}
STARTERS_ONLY = True              # builder player legs: only players who started at least 3 of their last 5 games (needs data/player_context_latest.json)
PCTX = {}
W_OURS = 0.35                     # share of OUR team model when a market total/spread exists (rest is the bookmakers'); provisional, same as the player-prop blend
PACE_BAND = 4                     # market-blended total this far below / above the league average = slow / fast game
TT_OVER_GAP, TT_UNDER_GAP = 20, 20  # your bookie's team-total alternates reach 20 points below (Over) and 20 above (Under) the main team total
MIN_MINUTES = 20                  # players projected under this many minutes are not used for legs
TT_STEP = 1                       # team-total table: a rung every point, from main-20 to main+20 (your bookie goes up in 1s)
TT_GAP = 20                       # same 20-point reach when there is no market team total (measured from our own projection)
TOT_GAP = 10                      # your bookie's alternate GAME totals reach 10 points below (Over) and 10 above (Under) the main total
SPREAD_GAP = 7                    # highest alternate spread = the team's own spread + 7
MIN_PLUS_CAP = 1.0                # a heavy favourite (own spread + 7 below this) gets no plus-spread leg at all
MIN_SHOW = 2                       # always build at least this many legs
# how strongly a player's stat follows his team's score and drops in blowouts. Points and PRA were measured on two seasons;
# the others are careful guesses (smaller, because rebounds and assists depend less on team scoring) and have not been measured yet
RHO = {'pts': 0.23, 'pra': 0.25, 'reb': 0.10, 'ast': 0.15, 'pr': 0.20, 'pa': 0.22, 'ar': 0.12}
BLOW = {'pts': -0.09, 'pra': -0.115, 'reb': -0.05, 'ast': -0.05, 'pr': -0.08, 'pa': -0.08, 'ar': -0.06}


def Phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


PhiV = np.vectorize(Phi)
_erf = np.vectorize(erf)


def ndtr(z):
    return 0.5 * (1 + _erf(z / sqrt(2)))


def fmt_date(iso):
    y, m, d = iso[:10].split('-')
    return f'{d}/{m}/{y}'


# ------------------------------------------------------------------ team games
def load_games(path=None):
    path = path or os.path.join(DATA, 'player_games.csv')
    d = pd.read_csv(path, low_memory=False)
    d = d[(d.dnp == 0) & (d.season_type.isin([2, 3, 5]))]
    g = d.groupby(['event_id', 'team']).agg(
        date=('game_date_et', 'first'), sy=('season_year', 'first'), home=('home', 'first'),
        pts=('pts', 'sum'), mn=('min', 'sum'), fga=('fga', 'sum'), fta=('fta', 'sum'),
        oreb=('oreb', 'sum'), tov=('tov', 'sum')).reset_index()
    g['poss'] = g.fga + 0.44 * g.fta - g.oreb + g.tov
    g = g[(g.mn >= 235) & (g.pts >= 60)]
    h = g[g.home == 1].set_index('event_id')
    a = g[g.home == 0].set_index('event_id')
    j = h.join(a, lsuffix='_h', rsuffix='_a', how='inner').reset_index()
    j = j.rename(columns={'date_h': 'date', 'sy_h': 'sy'})
    j['min_g'] = ((j.mn_h + j.mn_a) / 2) / 5
    j['poss'] = (j.poss_h + j.poss_a) / 2 * 48 / j.min_g
    j = j.sort_values(['date', 'event_id']).reset_index(drop=True)
    return j[['event_id', 'date', 'sy', 'team_h', 'team_a', 'pts_h', 'pts_a', 'poss', 'min_g']]


class Ratings:
    def __init__(self):
        self.off, self.de, self.pace, self.n = {}, {}, {}, {}
        self.lg_ppp, self.lg_pace, self.cur = 110.0, 100.0, None

    def _new_season(self, sy):
        if self.cur is not None and sy != self.cur:
            for t in self.off:
                self.off[t] *= CARRY
                self.de[t] *= CARRY
                self.pace[t] *= PCARRY
                self.n[t] = 0
        self.cur = sy

    def _ensure(self, t):
        if t not in self.off:
            self.off[t] = self.de[t] = self.pace[t] = 0.0
            self.n[t] = 0

    def predict(self, H, A, c=1.0):
        self._ensure(H)
        self._ensure(A)
        poss = self.lg_pace + self.pace[H] + self.pace[A]
        ph = self.lg_ppp + self.off[H] + self.de[A] + HCA / 2
        pa = self.lg_ppp + self.off[A] + self.de[H] - HCA / 2
        return poss * ph / 100 * c, poss * pa / 100 * c, poss, ph, pa

    def update(self, r):
        self._new_season(r.sy)
        H, A = r.team_h, r.team_a
        _, _, poss_p, ph, pa = self.predict(H, A)
        mg = r.min_g / 48
        pph = r.pts_h / (r.poss * mg) * 100
        ppa = r.pts_a / (r.poss * mg) * 100
        for t, o, rr in ((H, A, pph - ph), (A, H, ppa - pa)):
            self.off[t] += max(LR, 1 / (self.n[t] + K0)) * rr
            self.de[o] += max(LR, 1 / (self.n[o] + K0)) * rr
        rp = r.poss - poss_p
        for t in (H, A):
            self.pace[t] += max(PLR, 1 / (self.n[t] + K0)) * rp
            self.n[t] += 1
        self.lg_ppp += 0.002 * ((pph + ppa) / 2 - self.lg_ppp)
        self.lg_pace += 0.002 * (r.poss - self.lg_pace)


def walk(j):
    """Walk through games in date order; record the prediction made BEFORE each game."""
    R = Ratings()
    rows = []
    for r in j.itertuples():
        R._new_season(r.sy)
        mh, ma, _, _, _ = R.predict(r.team_h, r.team_a)
        rows.append((r.event_id, r.date, r.sy, r.team_h, r.team_a, mh, ma, r.pts_h, r.pts_a))
        R.update(r)
    return R, pd.DataFrame(rows, columns=['event_id', 'date', 'sy', 'H', 'A', 'mh', 'ma', 'ph', 'pa'])


def calibrate(o):
    f = o[(o.sy == FIT_SEASON) & (o.date < FIT_END)]
    c = (f.ph + f.pa).sum() / (f.mh + f.ma).sum()
    f2 = o[o.sy == FIT_SEASON]
    mu_m = f2.mh - f2.ma
    mu_t = (f2.mh + f2.ma) * c
    sm = float((f2.ph - f2.pa - mu_m).std())
    st = float((f2.ph + f2.pa - mu_t).std())
    return c, sm, st


# ------------------------------------------------------------------ recent form + injuries
FORM_N = 10          # last 10 games
FORM_COLS = ['own_form', 'opp_def_form', 'own_miss', 'opp_miss']
USE_FORM = True      # switched on because the backtest (data/form_backtest.json) shows it predicts team points better


def form_table(o, c):
    """One row per team-game: how far the model was off (res), plus what was known BEFORE the game:
    own_form = how much the team scored vs the model over its last 10, opp_def_form = how much the opponent allowed vs the model over its last 10,
    own_miss / opp_miss = points per game of rotation players (15+ min) missing from the lineup."""
    rows = []
    for r in o.itertuples():
        rows.append((r.event_id, r.date, r.sy, r.H, r.A, r.mh * c, r.ph))
        rows.append((r.event_id, r.date, r.sy, r.A, r.H, r.ma * c, r.pa))
    L = pd.DataFrame(rows, columns=['ev', 'date', 'sy', 'team', 'opp', 'pred', 'act']).sort_values(['date', 'ev']).reset_index(drop=True)
    L['res'] = L.act - L.pred
    L['own_form'] = L.groupby('team').res.transform(lambda x: x.shift(1).rolling(FORM_N, min_periods=5).mean())
    d = L[['ev', 'date', 'opp', 'res']].rename(columns={'opp': 'team', 'res': 'allowed'}).sort_values(['date', 'ev'])
    d['opp_def_form'] = d.groupby('team').allowed.transform(lambda x: x.shift(1).rolling(FORM_N, min_periods=5).mean())
    L = L.merge(d[['ev', 'team', 'opp_def_form']].rename(columns={'team': 'opp'}), on=['ev', 'opp'], how='left')
    pg = pd.read_csv(os.path.join(DATA, 'player_games.csv'), dtype={'athlete_id': str}, low_memory=False)
    pg = pg[pg.season_type.isin([2, 3, 5]) & (pg.dnp == 0) & (pg['min'] > 0)].sort_values('game_date_et')
    play_sets = pg.groupby(['event_id', 'team']).athlete_id.apply(set).to_dict()
    miss = {}
    for team, gt in L.groupby('team'):
        tp = pg[pg.team == team]
        byev = {e: g for e, g in tp.groupby('event_id')}
        hist = []
        for e in gt.sort_values(['date', 'ev']).ev:
            recent = hist[-FORM_N:]
            if len(recent) >= 5:
                tot = {}
                for gd_ in recent:
                    for a, (pt, mn) in gd_.items():
                        t = tot.setdefault(a, [0.0, 0.0])
                        t[0] += pt
                        t[1] += mn
                ps = play_sets.get((e, team), set())
                miss[(e, team)] = sum(v[0] / len(recent) for a, v in tot.items() if v[1] / len(recent) >= 15 and a not in ps)
            g = byev.get(e)
            hist.append({r.athlete_id: (r.pts, getattr(r, 'min')) for r in g.itertuples()} if g is not None else {})
    L['own_miss'] = [miss.get((e, t), np.nan) for e, t in zip(L.ev, L.team)]
    L['opp_miss'] = [miss.get((e, t), np.nan) for e, t in zip(L.ev, L.opp)]
    return L, pg


def fit_form(tr):
    X = np.column_stack([np.ones(len(tr))] + [tr[k] for k in FORM_COLS])
    return np.linalg.lstsq(X, tr.res.values, rcond=None)[0]


def form_backtest(o, c):
    """Does the adjustment predict a team's points better on games it has not seen? Fit before 15/01/2026, test after (plus this season)."""
    L, _ = form_table(o, c)
    L = L.dropna(subset=['own_form', 'opp_def_form', 'own_miss', 'opp_miss'])
    tr = L[(L.sy < FIT_SEASON) | ((L.sy == FIT_SEASON) & (L.date < FIT_END))]
    te = L[~L.index.isin(tr.index)]
    b = fit_form(tr)
    Xt = np.column_stack([np.ones(len(te))] + [te[k] for k in FORM_COLS])
    e1 = te.res.values - Xt @ b
    e0 = te.res.values
    gain = e0 ** 2 - e1 ** 2
    out = {'train_team_games': int(len(tr)), 'test_team_games': int(len(te)),
           'rmse_without': round(float(np.sqrt((e0 ** 2).mean())), 3), 'rmse_with': round(float(np.sqrt((e1 ** 2).mean())), 3),
           'squared_error_gain_t': round(float(gain.mean() / (gain.std() / np.sqrt(len(gain)))), 2),
           'coefficients': dict(zip(['intercept'] + FORM_COLS, [round(float(x), 3) for x in b]))}
    with open(os.path.join(DATA, 'form_backtest.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def live_form(L, pg, proj_players, team, opp):
    """Features for an upcoming game: recent form from the table, and expected missing scoring from each rotation player's chance of playing."""
    own_form = float(L[L.team == team].res.tail(FORM_N).mean()) if (L.team == team).any() else 0.0
    opp_def = float(L[L.opp == opp].res.tail(FORM_N).mean()) if (L.opp == opp).any() else 0.0

    def expected_miss(tm):
        tp = pg[pg.team == tm]
        evs = list(dict.fromkeys(tp.sort_values('game_date_et').event_id))[-FORM_N:]
        if len(evs) < 5:
            return 0.0
        w = tp[tp.event_id.isin(evs)]
        agg = w.groupby('athlete_id').agg(pts=('pts', 'sum'), mn=('min', 'sum'))
        pp = {p['id']: p['p_play'] for p in proj_players if p['team'] == tm}
        tot = 0.0
        for a, r in agg.iterrows():
            if r.mn / len(evs) >= 15:
                tot += (1.0 - pp.get(a, 0.0)) * r.pts / len(evs)
        return tot
    return [own_form, opp_def, expected_miss(team), expected_miss(opp)]


# ------------------------------------------------------------------ backtest
def backtest():
    j = load_games()
    R, o = walk(j)
    c, sm_all, st_all = calibrate(o)
    fit = o[(o.sy == FIT_SEASON) & (o.date < FIT_END)]
    chk = o[(o.sy == FIT_SEASON) & (o.date >= FIT_END)].copy()
    sm = float(((fit.ph - fit.pa) - (fit.mh - fit.ma)).std())
    st = float(((fit.ph + fit.pa) - (fit.mh + fit.ma) * c).std())
    chk['mu'] = chk.mh - chk.ma
    chk['tt'] = (chk.mh + chk.ma) * c
    chk['mar'] = chk.ph - chk.pa
    chk['tot'] = chk.ph + chk.pa
    out = {'run': datetime.datetime.now(GMT8).strftime('%d/%m/%Y %H:%M GMT+8'),
           'check_games': int(len(chk)), 'sigma_margin_fit': round(sm, 2), 'sigma_total_fit': round(st, 2),
           'margin_rmse_check': round(float(((chk.mar - chk.mu) ** 2).mean() ** .5), 2),
           'total_rmse_check': round(float(((chk.tot - chk.tt) ** 2).mean() ** .5), 2),
           'total_bias_check': round(float((chk.tot - chk.tt).mean()), 2)}
    # baselines: season-to-date margin diff, shrunk by 10 games
    j2 = j.copy()
    j2['mar'] = j2.pts_h - j2.pts_a
    sd = []
    hist = {}
    for r in j2.itertuples():
        def avg(t):
            s = [x for x in hist.get(t, []) if x[0] == r.sy]
            return sum(x[1] for x in s) / (len(s) + 10)
        sd.append(avg(r.team_h) - avg(r.team_a))
        hist.setdefault(r.team_h, []).append((r.sy, r.mar))
        hist.setdefault(r.team_a, []).append((r.sy, -r.mar))
    j2['sd'] = sd
    F = j2[(j2.sy == FIT_SEASON) & (j2.date < FIT_END)]
    C = j2[(j2.sy == FIT_SEASON) & (j2.date >= FIT_END)]
    b = np.polyfit(F.sd, F.mar, 1)
    out['baseline_margin_rmse_check'] = round(float(((C.mar - (b[1] + b[0] * C.sd)) ** 2).mean() ** .5), 2)
    # win chance reliability
    p = PhiV(chk.mu / sm)
    y = (chk.mar > 0).astype(float)
    out['win_brier'] = round(float(((p - y) ** 2).mean()), 4)
    rel = []
    for lo, hi in ((0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, 1.01)):
        k = (p >= lo) & (p < hi)
        if k.sum():
            rel.append({'range': f'{lo:.2f}-{min(hi, 1):.2f}', 'n': int(k.sum()), 'ours': round(float(p[k].mean()), 3),
                        'actual': round(float(y[k].mean()), 3)})
    out['win_reliability'] = rel
    # team +L lines
    lines = []
    for tgt in (0.80, 0.88, 0.94):
        P, Y = [], []
        for L in np.arange(0.5, 30.5, 1.0):
            for sign in (1, -1):
                mu = sign * chk.mu
                m = sign * chk.mar
                pr = PhiV((mu + L) / sm)
                k = (pr >= tgt - 0.02) & (pr < tgt + 0.02)
                P += list(pr[k])
                Y += list((m > -L)[k])
        if P:
            lines.append({'chance_wanted': tgt, 'legs': len(P), 'ours': round(float(np.mean(P)), 3),
                          'actual': round(float(np.mean(Y)), 3)})
    out['team_plus_line_reliability'] = lines
    tot = []
    for off in (-15, -10, -5, 0, 5, 10, 15):
        pr = 1 - PhiV(np.full(len(chk), off) / st)
        tot.append({'over_line_vs_ours': off, 'ours': round(float(pr.mean()), 3),
                    'actual': round(float((chk.tot > chk.tt + off).mean()), 3)})
    out['total_over_reliability'] = tot
    # single-team total (over a team's own points), offsets from our own team mean, -20 .. +20 like your bookie's ladder
    sdt = ((sm ** 2 + st ** 2) / 4) ** .5
    mh_ = chk.tt / 2 + chk.mu / 2
    ma_ = chk.tt / 2 - chk.mu / 2
    tm_rows = []
    for off in range(-20, 21, 5):
        pr = 1 - PhiV(np.full(len(chk) * 2, off) / sdt)
        act = np.concatenate([(chk.ph > mh_ + off).values, (chk.pa > ma_ + off).values]).astype(float)
        tm_rows.append({'over_line_vs_our_team_mean': off, 'ours': round(float(pr.mean()), 3), 'actual': round(float(act.mean()), 3)})
    out['team_total_over_reliability'] = tm_rows
    out['team_points_sd_used'] = round(float(sdt), 2)
    out['team_points_sd_actual'] = round(float(np.concatenate([(chk.ph - mh_).values, (chk.pa - ma_).values]).std()), 2)
    form_backtest(o, c)
    with open(os.path.join(DATA, 'team_backtest.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ builder
def load_game_lines():
    """Market total / spread per game: {(home, away): [records with date, total, home_spread]}. Manual file wins over the feed."""
    out = {}
    try:
        for g in json.load(open(os.path.join(DATA, 'game_lines_latest.json'))).get('games', {}).values():
            out.setdefault((g['home'], g['away']), []).append(dict(g, src='feed'))
    except Exception:
        pass
    try:
        with open(os.path.join(DATA, 'game_lines_manual.csv'), newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                rec = {'date': r['date'].strip(), 'src': 'manual'}
                for k in ('total', 'home_spread'):
                    if (r.get(k) or '').strip():
                        rec[k] = float(r[k])
                out.setdefault((r['home'].strip(), r['away'].strip()), []).insert(0, rec)
    except Exception:
        pass
    return out


def pick_market(table, home, away, date):
    best = None
    for rec in table.get((home, away), []):
        try:
            d = abs((datetime.datetime.strptime(rec['date'], '%d/%m/%Y') - datetime.datetime.strptime(date, '%d/%m/%Y')).days)
        except Exception:
            continue
        if d <= 1 and (best is None or rec['src'] == 'manual'):
            best = rec
    return best


def load_lines():
    try:
        L = json.load(open(os.path.join(DATA, 'lines_latest.json')))
    except Exception:
        return {}
    out = {}
    for p in L.get('players', []):
        out[(p['n'], p['t'], p['d'])] = p
    return out


def anchor(pl, line_rec, stat):
    """Main line used as the anchor: the higher of the DK/FD main lines (so the rung surely exists), else our median (flagged est).
    Only points and PRA have book lines in the feed; every other stat is an estimate."""
    est = True
    main = None
    books = ''
    if line_rec and line_rec.get(stat) and line_rec[stat].get('m'):
        ls = [b[1] for b in line_rec[stat]['m'] if b[1] is not None]
        if ls:
            main = round(float(max(ls)) * 2) / 2
            est = False
            books = ' / '.join(f"{BOOK.get(b[0], b[0][:2].upper())} {b[1]:g}" for b in line_rec[stat]['m'])
    if main is None:
        main = math.floor(pl[stat]['p50']) + 0.5
        books = 'est. from our median'
    return main, est, books


def k_floor(stat, main, est, k_pts):
    """Lowest rung we assume the bookie offers for this stat."""
    if stat == 'pts':
        return max(1, int(math.floor(main) + 1 - PTS_GAP))
    if stat in ('reb', 'ast'):
        return FLOOR_RA
    if stat == 'ar':
        return 2 * FLOOR_RA
    if stat in ('pr', 'pa'):
        return k_pts + FLOOR_RA
    if stat == 'pra':
        return max(1, int(math.floor(main) + 1 - PRA_GAP)) if not est else k_pts + 2 * FLOOR_RA
    return 1


def cdf_draw(cdf, u):
    return np.searchsorted(np.asarray(cdf), u)


def simulate_game(g, sm, st, c, players, lines, rng):
    n = N_SIM
    mu_m = g['mh'] - g['ma']
    mu_t = (g['mh'] + g['ma']) * c
    mk = g.get('market') or {}
    mk_t = mk.get('total')
    mk_m = -mk['home_spread'] if mk.get('home_spread') is not None else None
    if mk_t is not None:
        mu_t = W_OURS * mu_t + (1 - W_OURS) * mk_t
    if mk_m is not None:
        mu_m = W_OURS * mu_m + (1 - W_OURS) * mk_m
    lg = g.get('lg_total')
    pace = 'normal'
    if lg:
        pace = 'slow' if mu_t <= lg - PACE_BAND else 'fast' if mu_t >= lg + PACE_BAND else 'normal'
    ctx = ''
    if mk_t is not None or mk_m is not None:
        ctx = 'market ' + (f'total {mk_t:g}' if mk_t is not None else '') + (' ' if mk_t is not None and mk_m is not None else '') + \
              (f'{g["home"]} {-mk_m:+g}' if mk_m is not None else '') + f' \u00b7 {pace} game'
    M = rng.normal(mu_m, sm, n)
    T = rng.normal(mu_t, st, n)
    PH, PA = (T + M) / 2, (T - M) / 2
    sdt = sqrt((sm ** 2 + st ** 2) / 4)
    zH, zA = (PH - mu_t / 2 - mu_m / 2) / sdt, (PA - mu_t / 2 + mu_m / 2) / sdt
    zb = (np.abs(M) - np.abs(M).mean()) / 10
    cands = []   # dict(label, kind, family, hit(bool array), p, note)

    def add(label, family, hit, note='', kind='team'):
        cands.append({'label': label, 'family': family, 'hit': hit, 'p': float(hit.mean()), 'note': note, 'kind': kind})

    H, A = g['home'], g['away']
    # spreads (team plus / minus), one rung per target chance
    anc_m = mk_m if mk_m is not None else mu_m
    for tm, mg, tmu in ((H, M, anc_m), (A, -M, -anc_m)):
        cap = round((SPREAD_GAP - tmu) * 2) / 2
        if cap < MIN_PLUS_CAP:
            continue
        for tgt in TEAM_TARGETS:
            for L in list(np.arange(-15.5, cap - 1e-9, 1.0)) + [cap]:      # team margin > -L  (L>0: team +L)
                hit = mg > -L
                if hit.mean() >= tgt or L >= cap - 1e-9:
                    lab = f'{tm} +{L:g}' if L > 0 else f'{tm} {L:g}'
                    add(lab, f'spread_{tm}', hit, ctx)
                    break
    # moneyline
    for tm, mg in ((H, M), (A, -M)):
        add(f'{tm} to win', f'spread_{tm}', mg > 0, 'moneyline' + (' \u00b7 ' + ctx if ctx else ''))
    # game total
    anc_t = mk_t if mk_t is not None else mu_t
    for tgt in TEAM_TARGETS:
        if pace != 'slow':
            lo = math.floor(anc_t - TOT_GAP) + 0.5
            for xx in np.arange(lo, mu_t + 40, 1.0):
                if (T > xx).mean() <= tgt + 1e-9:
                    add(f'Game total over {xx:g}', 'total', T > xx, ctx)
                    break
        if pace != 'fast':
            hi = math.floor(anc_t + TOT_GAP) + 0.5
            for xx in np.arange(hi, mu_t - 40, -1.0):
                if (T < xx).mean() <= tgt + 1e-9:
                    add(f'Game total under {xx:g}', 'total', T < xx, ctx)
                    break
    # players
    singles = []
    for pl in players:
        pp = pl['p_play']
        if pp < 0.9 or pl['minutes']['mean'] < MIN_MINUTES:
            continue
        if STARTERS_ONLY and PCTX and not (PCTX.get(f"{pl['id']}|{pl['event_id']}") or {}).get('st'):
            continue
        zt = zH if pl['team'] == H else zA
        # Leg chances are "given he plays" (bookies void / refund a leg if the player does not play, and p_play >= 0.9 here anyway).
        # Multiplying by p_play used to shave up to 10% off every leg and made the model odds look too long next to the bookie's.
        plays = np.ones(n, bool)
        e = rng.normal(0, 1, n)
        lrec = lines.get((pl['name'], pl['team'], pl['game_date']))
        if 'pts' not in pl:
            continue
        main_p, est_p, _ = anchor(pl, lrec, 'pts')
        k_pts = max(1, int(math.floor(main_p) + 1 - PTS_GAP))     # points floor, also the base for the combos
        best = None
        for stat in STATS:
            if stat not in pl:
                continue
            a, b = RHO[stat], BLOW[stat]
            s = sqrt(max(0.5, 1 - a * a - b * b))
            z = a * zt + b * zb + s * e
            u = ndtr(z)
            x = cdf_draw(pl[stat]['cdf'], u)
            main, est, books = anchor(pl, lrec, stat)
            if main < (MIN_MAIN_EST if est else MIN_MAIN)[stat]:
                continue
            k_low = k_floor(stat, main, est, k_pts)               # lowest rung we assume the bookie could offer
            # walk up your bookie's ladder from the lowest rung and keep the one whose chance lands closest to the target price (inside this stat's band)
            lo, hi = BAND[stat]
            rungs = [r for r in LADDER[stat] if r <= int(math.floor(main))]
            below = [r for r in rungs if r <= k_low]
            ks = [r for r in rungs if r >= (below[-1] if below else rungs[0])] if rungs else []   # start one rung at or below the lowest assumed rung
            opts = []
            for k in ks:
                pk = float((plays & (x >= k)).mean())
                if pk < lo:
                    break
                if pk <= hi:
                    opts.append((k, pk))
            pick = None
            if opts:
                pick = max(opts, key=lambda kq: kq[1])   # highest chance of hitting inside the 1.20-1.80 window
            nm = STAT_NAME[stat]
            if pick is not None:
                k_min = pick[0]
                hit = plays & (x >= k_min)
                cand = {'label': f"{pl['name']} {k_min}+ {nm}", 'family': f"pl_{pl['id']}", 'hit': hit, 'p': float(hit.mean()),
                        'note': ('line est. (no DK/FD line)' if est else f'main {main:g} ({books})'), 'kind': 'player',
                        'team': pl['team'], 'stat': stat, 'est': est, 'k': k_min, 'main': main, 'pid': pl['id']}
                if best is None or cand['p'] > best['p']:
                    best = cand
            if not est:
                pu = float((plays & (x <= math.floor(main))).mean())   # Under needs the player to play too (conservative)
                singles.append({'label': f"{pl['name']} UNDER {main:g} {nm}", 'p': pu, 'note': f'main line only ({books})',
                                'hit': plays & (x <= math.floor(main)), 'player': pl['name']})
        if best:
            cands.append(best)
    # team-total legs: for each team, the Over and the Under (any line from main-20 to main+20, steps of 1) with the HIGHEST chance inside the odds window.
    # The main line is the DK/FD team total when the feed has it, otherwise our estimate.
    for tm, P, mu, sign in ((H, PH, mu_t / 2 + mu_m / 2, 1), (A, PA, mu_t / 2 - mu_m / 2, -1)):
        has = mk_t is not None and mk_m is not None
        real = mk.get('home_tt' if sign == 1 else 'away_tt')
        anc = real if real is not None else (mk_t / 2 + sign * mk_m / 2) if has else mu
        main_tt = float(real) if real is not None else math.floor(anc) + 0.5
        note = f'main {main_tt:g}' + ('' if real is not None else ' (estimated)') + (' \u00b7 ' + pace + ' game' if pace != 'normal' else '')
        for side in ('over', 'under'):
            if (side == 'over' and pace == 'slow') or (side == 'under' and pace == 'fast'):
                continue
            best = None
            for d in range(-TT_GAP, TT_GAP + 1, TT_STEP):
                x = math.floor(main_tt) + 0.5 + d      # always half-point lines
                hit = (P > x) if side == 'over' else (P < x)
                pr = float(hit.mean())
                if LEG_LO <= pr <= LEG_HI and (best is None or pr > best[1]):
                    best = (x, pr, hit)
            if best:
                add(f'{tm} team total {side} {best[0]:g}', f'tt_{tm}', best[2], note)
    return cands, singles, dict( M=M, T=T, PH=PH, PA=PA, mu_t=mu_t, mu_m=mu_m, pace=pace, ctx=ctx)


def pick_pool(cands, singles):
    # unders at the main line may join the pool when strong
    for s in singles:
        if s['p'] >= MIN_LEG:
            cands.append({'label': s['label'], 'family': 'pl_u_' + s['player'], 'hit': s['hit'], 'p': s['p'],
                          'note': s['note'], 'kind': 'player_under'})
    def in_band(c):
        lo, hi = BAND[c['stat']] if c.get('stat') in BAND else (LEG_LO, PLAYER_HI) if c.get('kind') == 'player_under' else (LEG_LO, LEG_HI)
        return lo <= c['p'] <= hi
    pool = [c for c in cands if in_band(c)]
    if not pool:
        return [], 0.0, []
    # every game gets at least MIN_PLAYER_LEGS player props plus its team-total legs (one per team), highest chance first; 5 legs at most
    def score(c):
        return c['p'] + (PLAYER_BONUS if c['kind'] != 'team' else 0.0)
    chosen, used = [], set()
    players_ = sorted([c for c in pool if c['kind'] == 'player'], key=score, reverse=True)
    teams_ = sorted([c for c in pool if c['kind'] == 'team' and str(c['family']).startswith('tt_')], key=score, reverse=True)
    for c in players_:
        if sum(1 for x in chosen if x['kind'] != 'team') >= MIN_PLAYER_LEGS:
            break
        if c['family'] not in used:
            chosen.append(c)
            used.add(c['family'])
    for c in teams_:
        if sum(1 for x in chosen if x['kind'] == 'team') >= MAX_TEAM_LEGS or len(chosen) >= MAX_LEGS:
            break
        if c['family'] not in used:
            chosen.append(c)
            used.add(c['family'])
    if not chosen:
        return [], 0.0, []
    allhit = np.ones(len(chosen[0]['hit']), bool)
    for c in chosen:
        allhit &= c['hit']
    drop = []
    for i, c in enumerate(chosen):
        w = np.ones(len(allhit), bool)
        for k, d in enumerate(chosen):
            if k != i:
                w &= d['hit']
        drop.append(round(float(w.mean()), 3))
    return chosen, float(allhit.mean()), drop


def project():
    t0 = time.time()
    j = load_games()
    R, o = walk(j)
    c, sm, st = calibrate(o)
    last = j.sy.max()
    R._new_season(last)
    proj = json.load(open(os.path.join(DATA, 'projections_latest.json')))
    by_ev = {}
    for p in proj['players']:
        by_ev.setdefault(p['event_id'], []).append(p)
    lines = load_lines()
    global PCTX
    try:
        PCTX = json.load(open(os.path.join(DATA, 'player_context_latest.json')))['players']
    except Exception:
        PCTX = {}
    gl = load_game_lines()
    FB, FPG = (None, None)
    if USE_FORM:
        FL, FPG = form_table(o, c)
        FB = fit_form(FL.dropna(subset=['own_form', 'opp_def_form', 'own_miss', 'opp_miss']))
    lg_total = float(j[j.sy == last].eval('pts_h + pts_a').mean())
    sched = pd.read_csv(os.path.join(DATA, 'schedule.csv'), dtype=str)
    sched = sched[(sched.state == 'pre') & (sched.season_type.isin(['2', '3', '5']))]
    rng = np.random.default_rng(7)
    games = []
    for r in sched.itertuples():
        H, A = r.home, r.away
        mh, ma, poss, _, _ = R.predict(H, A, c)
        pls = by_ev.get(r.event_id, [])
        if FB is not None:
            adj_h = float(FB[0] + np.dot(FB[1:], live_form(FL, FPG, pls, H, A)))
            adj_a = float(FB[0] + np.dot(FB[1:], live_form(FL, FPG, pls, A, H)))
            mh, ma = mh + adj_h, ma + adj_a
        g = {'event_id': r.event_id, 'home': H, 'away': A, 'mh': mh / c, 'ma': ma / c, 'lg_total': lg_total,
             'market': pick_market(gl, H, A, fmt_date(r.game_date_et))}
        players = by_ev.get(r.event_id, [])
        cands, singles, sim = simulate_game(g, sm, st, c, players, lines, rng)
        chosen, joint, drop = pick_pool(cands, singles)
        M, T = sim['M'], sim['T']
        pH = float((M > 0).mean())
        gd = {'event_id': r.event_id, 'game_date': fmt_date(r.game_date_et), 'tipoff_utc': r.tipoff_utc,
              'home': H, 'away': A,
              'proj': {'home_pts': round(sim['mu_t'] / 2 + sim['mu_m'] / 2, 1), 'away_pts': round(sim['mu_t'] / 2 - sim['mu_m'] / 2, 1),
                       'margin_home': round(sim['mu_m'], 1), 'total': round(sim['mu_t'], 1), 'pace': round(poss, 1), 'home_win': round(pH, 3),
                       'read': sim['pace'], 'market': g['market'] and {k: v for k, v in g['market'].items() if k in ('total', 'home_spread', 'src', 'home_tt', 'away_tt')}},
              'margin_lines': [{'team': H, 'line': -L, 'p': round(float((M > L).mean()), 3)} for L in (0.5, 5.5, 10.5)]
                              + [{'team': A, 'line': -L, 'p': round(float((-M > L).mean()), 3)} for L in (0.5, 5.5, 10.5)],
              'builder': {'legs': [{'label': x['label'], 'p': round(x['p'], 3), 'note': x['note'], 'kind': x['kind'], 'id': x.get('pid'), 'stat': x.get('stat')} for x in chosen],
                          'all_hit': round(joint, 3), 'without_leg': drop,
                          'players_in_play': len(players)},
              'unders': sorted([{'label': s['label'], 'p': round(s['p'], 3), 'note': s['note']} for s in singles if s['p'] >= 0.58],
                               key=lambda z: -z['p'])[:6]}
        games.append(gd)
    out = {'generated': datetime.datetime.now(GMT8).strftime('%d/%m/%Y %H:%M GMT+8'),
           'generated_epoch': int(time.time()),
           'sigma_margin': round(sm, 2), 'sigma_total': round(st, 2), 'total_scale': round(c, 4),
           'league_avg_total': round(lg_total, 1), 'games_with_market': sum(1 for x in games if x['proj']['market']),
           'settings': {'floor': FLOOR, 'leg_lo': LEG_LO, 'leg_hi': LEG_HI, 'pts_gap': PTS_GAP, 'pra_gap': PRA_GAP},
           'games': games}
    with open(os.path.join(DATA, 'games_latest.json'), 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'games {len(games)} in {time.time() - t0:.0f}s')
    for gd in games[:3]:
        print(json.dumps(gd, indent=1)[:2500])


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if mode in ('backtest', 'both'):
        backtest()
    if mode in ('project', 'both'):
        project()
