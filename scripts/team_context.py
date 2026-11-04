"""Team context for every upcoming game: form, ratings, schedule strength, home/road, rest, who is out.
Writes data/team_context_latest.json (read by index.html). Needs only data/player_games.csv, schedule.csv and the injury files.
Ratings are points per 100 possessions (possessions = FGA + 0.44*FTA - OREB + TOV, averaged over both teams). Rank 1 = best of 30."""
import csv, json, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lines import norm, TEAM_CODES  # noqa: E402

DATA, N, REG = Path("data"), 10, ("2", "3", "5")


def ordn(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rate(gs):
    pts, opp, poss = sum(g["pts"] for g in gs), sum(g["opp_pts"] for g in gs), sum(g["poss"] for g in gs)
    w = sum(g["pts"] > g["opp_pts"] for g in gs)
    return {"n": len(gs), "w": w, "l": len(gs) - w, "off": 100 * pts / poss, "def": 100 * opp / poss,
            "ppg": pts / len(gs), "opp_ppg": opp / len(gs), "pace": poss / len(gs)}


def ranks(vals, best_high):
    order = sorted(vals, key=lambda t: -vals[t] if best_high else vals[t])
    return {t: i + 1 for i, t in enumerate(order)}


def load():
    tg, plays = {}, defaultdict(list)
    with open(DATA / "player_games.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["season_type"] not in REG or r["dnp"] == "1":
                continue
            mn = float(r["min"] or 0)
            if mn <= 0:
                continue
            t = tg.setdefault((r["event_id"], r["team"]), {"ev": r["event_id"], "date": r["game_date_et"], "sy": int(r["season_year"]),
                              "team": r["team"], "opp": r["opp"], "home": r["home"] == "1", "pts": 0.0, "pr": 0.0})
            t["pts"] += float(r["pts"] or 0)
            t["pr"] += float(r["fga"] or 0) + 0.44 * float(r["fta"] or 0) - float(r["oreb"] or 0) + float(r["tov"] or 0)
            plays[norm(r["player"])].append((r["game_date_et"], r["team"], mn, float(r["pts"] or 0)))
    by = defaultdict(list)
    for (ev, team), t in tg.items():
        o = tg.get((ev, t["opp"]))
        if o:
            t["opp_pts"], t["poss"] = o["pts"], (t["pr"] + o["pr"]) / 2
            by[team].append(t)
    for v in by.values():
        v.sort(key=lambda g: (g["date"], g["ev"]))
    return by, plays


def injuries(date, team, plays):
    rows, src = {}, ""
    p = DATA / "injuries_official_latest.csv"
    if p.exists():
        for r in sorted(csv.DictReader(open(p, newline="", encoding="utf-8")), key=lambda r: r["report_utc"]):
            if r["game_date_et"] == date and r["team"] == team:
                rows[norm(r["player_norm"])], src = r["status"], "official report"
    if not rows and (DATA / "injuries_latest.csv").exists():
        for r in csv.DictReader(open(DATA / "injuries_latest.csv", newline="", encoding="utf-8")):
            if TEAM_CODES.get(r["team"]) == team:
                rows[norm(r["player"])], src = ("Out" if r["status"] == "Out" else "Questionable"), "ESPN list (no official report yet)"
    out = {"Out": [], "Doubtful": [], "Questionable": []}
    for nm, st in rows.items():
        g = [x for x in plays.get(nm, []) if x[1] == team][-N:]
        if st in out and len(g) >= 2 and sum(x[2] for x in g) / len(g) >= 12:
            out[st].append((nm.title(), sum(x[3] for x in g) / len(g), sum(x[2] for x in g) / len(g)))
    return out, src


def main():
    by, plays = load()
    cur = max(g["sy"] for v in by.values() for g in v)
    season = {t: [g for g in v if g["sy"] == cur] for t, v in by.items()}
    season = {t: v for t, v in season.items() if v}
    s_rt = {t: rate(v) for t, v in season.items()}
    l_rt = {t: rate(v[-N:]) for t, v in season.items()}
    s_off, s_def = ranks({t: r["off"] for t, r in s_rt.items()}, True), ranks({t: r["def"] for t, r in s_rt.items()}, False)
    l_off, l_def = ranks({t: r["off"] for t, r in l_rt.items()}, True), ranks({t: r["def"] for t, r in l_rt.items()}, False)
    avg_off = sum(r["off"] for r in l_rt.values()) / len(l_rt)
    avg_pace = sum(r["pace"] for r in l_rt.values()) / len(l_rt)

    def speed(pc):
        d = pc - avg_pace
        return "very fast" if d >= 3 else "fast" if d >= 1.5 else "slow" if d <= -1.5 and d > -3 else "very slow" if d <= -3 else "average"

    today = datetime.now(timezone.utc).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    sched = [r for r in csv.DictReader(open(DATA / "schedule.csv", newline="", encoding="utf-8")) if r["state"] == "pre" and r["game_date_et"] >= today] \
        or [r for r in csv.DictReader(open(DATA / "schedule.csv", newline="", encoding="utf-8")) if r["state"] == "pre"]
    games = {}
    for s in sched:
        card = {}
        for team, opp, home in ((s["home"], s["away"], True), (s["away"], s["home"], False)):
            if team not in season or opp not in season:
                continue
            L, last = l_rt[team], season[team][-N:]
            oo = [s_rt[g["opp"]] for g in last if g["opp"] in s_rt]
            adj_o = L["off"] - sum(x["def"] for x in oo) / len(oo)      # + = scored more than those defenses usually allow
            adj_d = sum(x["off"] for x in oo) / len(oo) - L["def"]      # + = allowed less than those offenses usually score
            od = sum(s_def[g["opp"]] for g in last if g["opp"] in s_def) / len(oo)
            of = sum(s_off[g["opp"]] for g in last if g["opp"] in s_off) / len(oo)
            sp = {h: rate([g for g in season[team] if g["home"] == h]) if any(g["home"] == h for g in season[team]) else None for h in (True, False)}
            sp_t = lambda r, nm: f"{nm} {r['w']}-{r['l']} (point margin {r['off'] - r['def']:+.1f} per 100 possessions)" if r else f"{nm}: no games yet"
            prev = [g for g in by[team] if g["date"] < s["game_date_et"]]
            gap = (datetime.fromisoformat(s["game_date_et"]) - datetime.fromisoformat(prev[-1]["date"])).days if prev else 9
            inj, src = injuries(s["game_date_et"], team, plays)
            fmt = lambda xs: ", ".join(f"{n} ({p:.0f} pts, {m:.0f} min)" for n, p, m in sorted(xs, key=lambda x: -x[1]))
            parts = [f"{k}: {fmt(v)}" for k, v in inj.items() if v]
            gone = sum(p for n, p, m in inj["Out"] + inj["Doubtful"] if m >= 15)
            card[("home" if home else "away")] = {"team": team, "lines": [
                f"Last {L['n']} games: {L['w']}-{L['l']} · scored {L['ppg']:.1f} points a game, allowed {L['opp_ppg']:.1f}"
                + (f" (only {L['n']} games played so far)" if L["n"] < N else ""),
                f"Speed: {L['pace']:.1f} possessions a game (a possession = one trip down the court) = {speed(L['pace'])} (league average {avg_pace:.1f})",
                f"Offense: {L['off']:.1f} points per 100 possessions = {ordn(l_off[team])} best of 30 (league average {avg_off:.1f})",
                f"Defense: lets in {L['def']:.1f} points per 100 possessions = {ordn(l_def[team])} best of 30 (lower is better)",
                f"Opponents faced: defenses averaged {ordn(round(od))}, offenses {ordn(round(of))} of 30. Scored {abs(adj_o):.1f} per 100 {'above' if adj_o > 0 else 'below'} "
                f"what those defenses usually allow; allowed {abs(adj_d):.1f} per 100 {'less' if adj_d > 0 else 'more'} than those offenses usually score",
                f"{sp_t(sp[True], 'Home')} · {sp_t(sp[False], 'Road')} (this season) · " + ("played yesterday (back-to-back)" if gap == 1 else f"{gap - 1} day{'' if gap == 2 else 's'} rest"),
                ("Injury list (" + src + "): " + " · ".join(parts) + (f" · about {gone:.0f} points a night of rotation scoring is out" if gone else "")) if parts
                else f"Injury list ({src or 'none yet'}): no rotation player listed"]}
        if len(card) == 2:
            h, a = s["home"], s["away"]
            card["matchup"] = (f"{a} offense {ordn(l_off[a])} vs {h} defense {ordn(l_def[h])} · {h} offense {ordn(l_off[h])} vs {a} defense {ordn(l_def[a])} (last {N} games, of 30)")
            games[s["event_id"]] = card
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "games": games}
    (DATA / "team_context_latest.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Team context for {len(games)} games")
    return 0


if __name__ == "__main__":
    sys.exit(main())
