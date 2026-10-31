"""Game model: team scores, margin, total, win chance, and the safest bet-builder pool per game.

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
PTS_GAP, PRA_GAP = 6, 12          # how far below the main line the bookie's alternate Overs go
TEAM_TARGETS = (0.85,)              # chance level tried for each team/game line (about the "Wizards +15" kind of leg)
MIN_LEG = 0.75                     # a leg must be at least this likely on its own
FLOOR = 0.45                       # pool keeps growing until the all-legs chance would drop below this; you pick the legs you want from it
MAX_LEGS = 6
MAX_TEAM_LEGS = 3                  # spread / total / team-total legs are one bet said three ways, so keep at most this many
MIN_LEG_PLAYER = 0.70              # player points legs at the bookie's lowest rung are rarely above 85%, so allow a bit lower
MAX_LEG_P = 0.97                   # skip legs so safe the bookie would pay almost nothing
MIN_MAIN = {'pts': 9.5, 'pra': 18.5}       # SMALLEST main points line that can be a builder leg (9.5 keeps Sarr 5+; lower it if your bookie offers legs on smaller lines)
MIN_MAIN_EST = {'pts': 9.5, 'pra': 24.5}   # same, for players with no DK/FD line (our own median is the anchor, flagged as an estimate)
BOOK = {'draftkings': 'DK', 'fanduel': 'FD'}
STATS = (('pts', PTS_GAP),)       # player legs: points only for now (PRA later)
W_OURS = 0.35                     # share of OUR team model when a market total/spread exists (rest is the bookmakers'); provisional, same as the player-prop blend
PACE_BAND = 4                     # market-blended total this far below / above the league average = slow / fast game
TT_OVER_GAP, TT_UNDER_GAP = 9, 12  # bookie's team-total alternates start ~9 below the market team total (WSH 108 -> 99.5) and reach ~12 above
MIN_MINUTES = 20                  # players projected under this many minutes are not used for legs
TT_GAP = 11                       # lowest alternate team-total Over is about this far below our projection (WSH 110.9 -> 99.5)
TOT_GAP = 14                      # same idea for game totals (a guess: tell me your bookie's real range)
SPREAD_GAP = 8                    # alternate spread goes about this far past our projected deficit (BOS -13 -> CHI +21.5)
MIN_SHOW = 3                       # always show at least this many legs
RHO = {'pts': 0.23, 'pra': 0.25}   # player points follow own team's score (measured both seasons)
BLOW = {'pts': -0.09, 'pra': -0.115}  # and drop in blowouts (per 10 pts of margin)


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
    """Main line used to place the alternate ladder: the higher of the DK/FD main lines (so the rung surely exists), else our median (flagged est)."""
    est = True
    main = None
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
        cap = math.floor(max(0.0, -tmu)) + SPREAD_GAP + 0.5   # bookie's highest alternate: about 8 past our projected deficit
        for tgt in TEAM_TARGETS:
            found = False
            for L in np.arange(-15.5, cap + 0.01, 1.0):      # team margin > -L  (L>0: team +L)
                hit = mg > -L
                if hit.mean() >= tgt or L >= cap - 1e-9:
                    lab = f'{tm} +{L:g}' if L > 0 else f'{tm} {L:g}'
                    add(lab, f'spread_{tm}', hit, ctx if hit.mean() >= tgt else ('highest alternate (our estimate of your bookie)' + (' \u00b7 ' + ctx if ctx else '')))
                    break
    # moneyline
    for tm, mg in ((H, M), (A, -M)):
        add(f'{tm} to win', f'spread_{tm}', mg > 0, 'moneyline' + (' \u00b7 ' + ctx if ctx else ''))
    # game total: Over from the lowest alternate up, Under from the highest alternate down.
    # Slow game (market total well under the league average): no Overs. Fast game: no Unders.
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
    # team totals, ladder placed from the market's own team total (total and spread) when we have it
    for tm, P, mu, sign in ((H, PH, mu_t / 2 + mu_m / 2, 1), (A, PA, mu_t / 2 - mu_m / 2, -1)):
        anc = (mk_t / 2 + sign * mk_m / 2) if (mk_t is not None and mk_m is not None) else mu
        o_gap, u_gap = (TT_OVER_GAP, TT_UNDER_GAP) if (mk_t is not None and mk_m is not None) else (TT_GAP, TT_GAP)
        for tgt in TEAM_TARGETS:
            if pace != 'slow':
                lo = math.floor(anc - o_gap) + 0.5
                for xx in np.arange(lo, mu + 30, 1.0):
                    if (P > xx).mean() <= tgt + 1e-9:
                        add(f'{tm} team total over {xx:g}', f'tt_{tm}', P > xx, ctx)
                        break
            if pace != 'fast':
                hi = math.floor(anc + u_gap) + 0.5
                for xx in np.arange(hi, mu - 30, -1.0):
                    if (P < xx).mean() <= tgt + 1e-9:
                        add(f'{tm} team total under {xx:g}', f'tt_{tm}', P < xx, ctx)
                        break
    # players
    singles = []
    for pl in players:
        pp = pl['p_play']
        if pp < 0.9 or pl['minutes']['mean'] < MIN_MINUTES:
            continue
        zt = zH if pl['team'] == H else zA
        plays = rng.random(n) < pp
        e = rng.normal(0, 1, n)
        best = None
        for stat, gap in STATS:
            a, b = RHO[stat], BLOW[stat]
            s = sqrt(max(0.5, 1 - a * a - b * b))
            z = a * zt + b * zb + s * e
            u = ndtr(z)
            x = cdf_draw(pl[stat]['cdf'], u)
            main, est, books = anchor(pl, lines.get((pl['name'], pl['team'], pl['game_date'])), stat)
            if main < (MIN_MAIN_EST if est else MIN_MAIN)[stat]:
                continue
            k_min = max(1, int(math.floor(main) + 1 - gap))       # lowest "N+" the bookie offers
            hit = plays & (x >= k_min)
            nm = 'pts' if stat == 'pts' else 'PRA'
            cand = {'label': f"{pl['name']} {k_min}+ {nm}", 'family': f"pl_{pl['id']}", 'hit': hit, 'p': float(hit.mean()),
                    'note': ('line est. (no DK/FD line)' if est else f'main {main:g} ({books})'), 'kind': 'player',
                    'team': pl['team'], 'stat': stat, 'est': est, 'k': k_min, 'main': main}
            if best is None or cand['p'] > best['p']:
                best = cand
            if not est:
                pu = float((plays & (x <= math.floor(main))).mean())   # Under needs the player to play too (conservative)
                singles.append({'label': f"{pl['name']} UNDER {main:g} {nm}", 'p': pu, 'note': f'main line only ({books})',
                                'hit': plays & (x <= math.floor(main)), 'player': pl['name']})
        if best:
            cands.append(best)
    return cands, singles, dict(M=M, T=T, PH=PH, PA=PA, mu_t=mu_t, mu_m=mu_m, pace=pace, ctx=ctx)


def pick_pool(cands, singles):
    # unders at the main line may join the pool when really strong
    for s in singles:
        if s['p'] >= MIN_LEG:
            cands.append({'label': s['label'], 'family': 'pl_u_' + s['player'], 'hit': s['hit'], 'p': s['p'],
                          'note': s['note'], 'kind': 'player_under'})
    pool = [c for c in cands if (MIN_LEG_PLAYER if c['kind'] == 'player' else MIN_LEG) <= c['p'] <= MAX_LEG_P]
    chosen, used = [], set()
    team_n = 0
    joint = np.ones(len(pool[0]['hit']), bool) if pool else None
    while pool and len(chosen) < MAX_LEGS:
        best = None
        for c in pool:
            if c['family'] in used:
                continue
            if c['kind'] == 'team' and team_n >= MAX_TEAM_LEGS:
                continue
            pj = float((joint & c['hit']).mean())
            if best is None or pj > best[0]:
                best = (pj, c)
        if best is None:
            break
        if best[0] < FLOOR and len(chosen) >= MIN_SHOW:
            break
        chosen.append(best[1])
        if best[1]['kind'] == 'team':
            team_n += 1
        used.add(best[1]['family'])
        joint = joint & best[1]['hit']
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
    gl = load_game_lines()
    lg_total = float(j[j.sy == last].eval('pts_h + pts_a').mean())
    sched = pd.read_csv(os.path.join(DATA, 'schedule.csv'), dtype=str)
    sched = sched[(sched.state == 'pre') & (sched.season_type.isin(['2', '3', '5']))]
    rng = np.random.default_rng(7)
    games = []
    for r in sched.itertuples():
        H, A = r.home, r.away
        mh, ma, poss, _, _ = R.predict(H, A, c)
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
                       'read': sim['pace'], 'market': g['market'] and {k: v for k, v in g['market'].items() if k in ('total', 'home_spread', 'src')}},
              'margin_lines': [{'team': H, 'line': -L, 'p': round(float((M > L).mean()), 3)} for L in (0.5, 5.5, 10.5)]
                              + [{'team': A, 'line': -L, 'p': round(float((-M > L).mean()), 3)} for L in (0.5, 5.5, 10.5)],
              'builder': {'legs': [{'label': x['label'], 'p': round(x['p'], 3), 'note': x['note'], 'kind': x['kind']} for x in chosen],
                          'all_hit': round(joint, 3), 'without_leg': drop,
                          'players_in_play': len(players)},
              'unders': sorted([{'label': s['label'], 'p': round(s['p'], 3), 'note': s['note']} for s in singles if s['p'] >= 0.58],
                               key=lambda z: -z['p'])[:6]}
        games.append(gd)
    out = {'generated': datetime.datetime.now(GMT8).strftime('%d/%m/%Y %H:%M GMT+8'),
           'generated_epoch': int(time.time()),
           'sigma_margin': round(sm, 2), 'sigma_total': round(st, 2), 'total_scale': round(c, 4),
           'league_avg_total': round(lg_total, 1), 'games_with_market': sum(1 for x in games if x['proj']['market']),
           'settings': {'floor': FLOOR, 'min_leg': MIN_LEG, 'pts_gap': PTS_GAP, 'pra_gap': PRA_GAP},
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
