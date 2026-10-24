"""Fetch NBA data from ESPN's public JSON and keep it as CSV files in data/.

Modes
  daily     schedule (next days), injuries snapshot, box scores for recent days
  backfill  box scores for a date range (resumable: games already saved are skipped)

Standard library only. Fails loudly: every error is logged, written to
data/health.json, and makes the run exit non-zero.
"""
import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ET = ZoneInfo("America/New_York")
DATA = Path("data")
PLAYER_GAMES = DATA / "player_games.csv"
SCHEDULE = DATA / "schedule.csv"
INJ_LATEST = DATA / "injuries_latest.csv"
INJ_HISTORY = DATA / "injuries_history.csv"
HEALTH = DATA / "health.json"

TIMEOUT = 25
PAUSE = 0.3  # seconds between requests, be polite
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
}

GAME_FIELDS = [
    "event_id", "game_date_et", "tipoff_utc", "season_year", "season_type",
    "team_id", "team", "opp_team_id", "opp", "home",
    "athlete_id", "player", "position", "starter", "dnp", "dnp_reason",
    "min", "pts", "reb", "ast", "oreb", "dreb", "stl", "blk", "tov", "pf",
    "fgm", "fga", "tpm", "tpa", "ftm", "fta", "plus_minus",
]
SCHEDULE_FIELDS = [
    "event_id", "game_date_et", "tipoff_utc", "state", "season_type",
    "home_id", "home", "away_id", "away",
]
INJURY_FIELDS = [
    "snapshot_utc", "team_id", "team", "athlete_id", "player", "status",
    "injury_date", "injury_type", "detail", "return_date", "short_comment",
]

ERRORS = []
WARNINGS = []


class SchemaError(Exception):
    pass


class NotFoundError(Exception):
    """ESPN answered 404: the page does not exist, retrying will not help."""


def log(msg):
    print(msg, flush=True)


def fetch_json(url, retries=4):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise NotFoundError(f"404 not found: {url}") from exc
            last = exc
        except Exception as exc:  # noqa: BLE001 - log everything, then retry
            last = exc
        wait = 2 ** attempt
        log(f"  retry {attempt + 1}/{retries} {url}: {type(last).__name__}: {last} (sleep {wait}s)")
        time.sleep(wait)
    raise RuntimeError(f"giving up on {url}: {type(last).__name__}: {last}")


# ---------- small parsing helpers ----------

def to_int(value, default=0):
    try:
        return int(str(value).replace("+", "").strip())
    except (ValueError, TypeError):
        return default


def parse_minutes(value):
    s = str(value).strip()
    try:
        if ":" in s:
            m, sec = s.split(":")
            return round(int(m) + int(sec) / 60, 2)
        return float(s)
    except (ValueError, TypeError):
        return 0.0


def made_att(value):
    try:
        a, b = str(value).split("-")
        return int(a), int(b)
    except (ValueError, TypeError):
        return 0, 0


def parse_utc(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def et_date_of(text):
    return parse_utc(text).astimezone(ET).date().isoformat()


# ---------- box scores ----------

def parse_box(d, event_id):
    """Turn one ESPN summary JSON into a list of player-game rows."""
    try:
        header = d["header"]
        comp = header["competitions"][0]
        competitors = {c["team"]["id"]: c for c in comp["competitors"]}
        tipoff = comp["date"]
        season = header.get("season") or {}
        team_blocks = d["boxscore"]["players"]
    except (KeyError, IndexError, TypeError) as exc:
        raise SchemaError(f"event {event_id}: missing key {exc!r}") from exc

    game_date = et_date_of(tipoff)
    rows = []
    team_points = {}

    for block in team_blocks:
        team = block["team"]
        tid = team["id"]
        mine = competitors.get(tid) or {}
        opp = next((c for k, c in competitors.items() if k != tid), {})
        for group in block.get("statistics", []):
            labels = group.get("labels") or []
            missing = [x for x in ("MIN", "PTS", "REB", "AST") if x not in labels]
            if missing:
                raise SchemaError(
                    f"event {event_id}: stat labels {labels} lack {missing}"
                )
            idx = {lab: i for i, lab in enumerate(labels)}
            for lab in ("FG", "3PT", "FT", "OREB", "DREB", "STL", "BLK", "TO", "PF", "+/-"):
                if lab not in idx:
                    WARNINGS.append(f"label {lab} missing in box score labels {labels}")

            for a in group.get("athletes", []):
                ath = a["athlete"]
                stats = a.get("stats") or []
                dnp = bool(a.get("didNotPlay")) or not stats

                def g(label):
                    i = idx.get(label)
                    return stats[i] if (i is not None and i < len(stats)) else ""

                fgm, fga = made_att(g("FG"))
                tpm, tpa = made_att(g("3PT"))
                ftm, fta = made_att(g("FT"))
                row = {
                    "event_id": event_id,
                    "game_date_et": game_date,
                    "tipoff_utc": tipoff,
                    "season_year": season.get("year", ""),
                    "season_type": season.get("type", ""),
                    "team_id": tid,
                    "team": team.get("abbreviation", ""),
                    "opp_team_id": (opp.get("team") or {}).get("id", ""),
                    "opp": (opp.get("team") or {}).get("abbreviation", ""),
                    "home": int(mine.get("homeAway") == "home"),
                    "athlete_id": ath.get("id", ""),
                    "player": ath.get("displayName", ""),
                    "position": (ath.get("position") or {}).get("abbreviation", ""),
                    "starter": int(bool(a.get("starter"))),
                    "dnp": int(dnp),
                    "dnp_reason": a.get("reason", "") or "",
                    "min": 0 if dnp else parse_minutes(g("MIN")),
                    "pts": 0 if dnp else to_int(g("PTS")),
                    "reb": 0 if dnp else to_int(g("REB")),
                    "ast": 0 if dnp else to_int(g("AST")),
                    "oreb": 0 if dnp else to_int(g("OREB")),
                    "dreb": 0 if dnp else to_int(g("DREB")),
                    "stl": 0 if dnp else to_int(g("STL")),
                    "blk": 0 if dnp else to_int(g("BLK")),
                    "tov": 0 if dnp else to_int(g("TO")),
                    "pf": 0 if dnp else to_int(g("PF")),
                    "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa,
                    "ftm": ftm, "fta": fta,
                    "plus_minus": 0 if dnp else to_int(g("+/-")),
                }
                if not row["athlete_id"] or not row["player"]:
                    # ESPN sometimes lists a player with only a short name.
                    # Keep the row (its minutes and points count for the team)
                    # under a stable made-up id, and warn.
                    short = ath.get("shortName") or ""
                    if not short:
                        raise SchemaError(f"event {event_id}: athlete without id or name: {ath}")
                    if not row["athlete_id"]:
                        row["athlete_id"] = f"unk-{row['team']}-{short}"
                    if not row["player"]:
                        row["player"] = short
                    WARNINGS.append(f"athlete without ESPN id kept as {row['athlete_id']}")
                rows.append(row)
                team_points[tid] = team_points.get(tid, 0) + row["pts"]

    # Sanity check: player points should add up to the team score.
    for tid, pts in team_points.items():
        score = to_int((competitors.get(tid) or {}).get("score"), default=-1)
        if score >= 0 and score != pts:
            WARNINGS.append(
                f"event {event_id}: team {tid} player pts {pts} != team score {score}"
            )
    return rows


def append_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerows(rows)


def rewrite_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def known_events():
    if not PLAYER_GAMES.exists():
        return set()
    with open(PLAYER_GAMES, newline="", encoding="utf-8") as f:
        return {r["event_id"] for r in csv.DictReader(f)}


def scoreboard_events(day):
    d = fetch_json(f"{BASE}/scoreboard?dates={day:%Y%m%d}&limit=100")
    return d.get("events", [])


def collect_boxscores(start, end):
    known = known_events()
    new_games = new_rows = 0
    day = start
    while day <= end:
        try:
            events = scoreboard_events(day)
        except Exception as exc:  # noqa: BLE001
            ERRORS.append(f"scoreboard {day}: {exc}")
            day += timedelta(days=1)
            continue
        done = [e for e in events if (e.get("status") or {}).get("type", {}).get("completed")]
        added = 0
        for ev in done:
            eid = str(ev["id"])
            if eid in known:
                continue
            time.sleep(PAUSE)
            try:
                rows = parse_box(fetch_json(f"{BASE}/summary?event={eid}"), eid)
                if not rows:
                    raise SchemaError(f"event {eid}: no player rows")
                append_rows(PLAYER_GAMES, GAME_FIELDS, rows)
                known.add(eid)
                added += 1
                new_rows += len(rows)
            except NotFoundError:
                WARNINGS.append(f"box score {eid} ({day}): ESPN has no summary (404), game skipped")
                log(f"  event {eid} ({day}): no summary on ESPN (404), skipped")
            except Exception as exc:  # noqa: BLE001
                ERRORS.append(f"box score {eid} ({day}): {type(exc).__name__}: {exc}")
        new_games += added
        log(f"{day}: {len(events)} events, {len(done)} completed, {added} new games saved")
        day += timedelta(days=1)
        time.sleep(PAUSE)
    return {"new_games": new_games, "new_rows": new_rows}


# ---------- schedule ----------

def collect_schedule(days_ahead):
    today = datetime.now(ET).date()
    rows = []
    for off in range(days_ahead + 1):
        day = today + timedelta(days=off)
        try:
            events = scoreboard_events(day)
        except Exception as exc:  # noqa: BLE001
            ERRORS.append(f"schedule {day}: {exc}")
            continue
        for ev in events:
            comp = (ev.get("competitions") or [{}])[0]
            by_side = {c.get("homeAway"): c for c in comp.get("competitors", [])}
            if not by_side:
                WARNINGS.append(f"schedule: event {ev.get('id')} has no competitors")
            home, away = by_side.get("home", {}), by_side.get("away", {})
            rows.append({
                "event_id": ev["id"],
                "game_date_et": et_date_of(ev["date"]),
                "tipoff_utc": ev["date"],
                "state": (ev.get("status") or {}).get("type", {}).get("state", ""),
                "season_type": (ev.get("season") or {}).get("type", ""),
                "home_id": (home.get("team") or {}).get("id", ""),
                "home": (home.get("team") or {}).get("abbreviation", ""),
                "away_id": (away.get("team") or {}).get("id", ""),
                "away": (away.get("team") or {}).get("abbreviation", ""),
            })
        time.sleep(PAUSE)
    if not rows:
        WARNINGS.append("schedule: no games found for the next days (offseason?)")
    rewrite_csv(SCHEDULE, SCHEDULE_FIELDS, rows)
    return {"schedule_rows": len(rows)}


# ---------- injuries ----------

def collect_injuries():
    d = fetch_json(f"{BASE}/injuries")
    snap = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for team in d.get("injuries", []):
        for inj in team.get("injuries", []):
            ath = inj.get("athlete") or {}
            kind = inj.get("type") or {}
            det = inj.get("details") or {}
            rows.append({
                "snapshot_utc": snap,
                "team_id": team.get("id", ""),
                "team": team.get("displayName", ""),
                "athlete_id": ath.get("id", ""),
                "player": ath.get("displayName", ""),
                "status": inj.get("status", ""),
                "injury_date": inj.get("date", ""),
                "injury_type": det.get("type") or kind.get("description") or kind.get("name") or "",
                "detail": det.get("detail", "") or "",
                "return_date": det.get("returnDate", "") or "",
                "short_comment": inj.get("shortComment", "") or "",
            })
    if not rows:
        WARNINGS.append("injuries: feed returned zero rows")
    else:
        no_id = sum(1 for r in rows if not r["athlete_id"])
        if no_id > 0.2 * len(rows):
            ERRORS.append(f"injuries: {no_id}/{len(rows)} rows lack athlete_id (schema changed?)")
    rewrite_csv(INJ_LATEST, INJURY_FIELDS, rows)
    append_rows(INJ_HISTORY, INJURY_FIELDS, rows)
    return {"injury_rows": len(rows)}


# ---------- health ----------

def data_summary():
    if not PLAYER_GAMES.exists():
        return {"games": 0, "player_rows": 0, "latest_game_date": None}
    games, n, latest = set(), 0, ""
    with open(PLAYER_GAMES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            games.add(r["event_id"])
            n += 1
            latest = max(latest, r["game_date_et"])
    return {"games": len(games), "player_rows": n, "latest_game_date": latest or None}


def write_health(mode, counts):
    HEALTH.parent.mkdir(parents=True, exist_ok=True)
    health = {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": mode,
        "ok": not ERRORS,
        "errors": ERRORS[:50],
        "error_count": len(ERRORS),
        "warnings": sorted(set(WARNINGS))[:20],
        "warning_count": len(WARNINGS),
        "this_run": counts,
        "data": data_summary(),
    }
    HEALTH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    return health


# ---------- main ----------

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["daily", "backfill"], required=True)
    p.add_argument("--start", help="backfill start date YYYY-MM-DD")
    p.add_argument("--end", help="backfill end date YYYY-MM-DD (default: today ET)")
    p.add_argument("--lookback-days", type=int, default=3)
    p.add_argument("--days-ahead", type=int, default=2)
    args = p.parse_args(argv)

    ERRORS.clear()
    WARNINGS.clear()
    today = datetime.now(ET).date()
    counts = {}
    try:
        if args.mode == "backfill":
            if not args.start:
                p.error("--start is required for backfill")
            start = date.fromisoformat(args.start)
            end = date.fromisoformat(args.end) if args.end else today
            log(f"Backfill box scores {start} to {end}")
            counts.update(collect_boxscores(start, end))
        else:
            log(f"Daily update, US Eastern today = {today}")
            try:
                counts.update(collect_injuries())
            except Exception as exc:  # noqa: BLE001
                ERRORS.append(f"injuries: {type(exc).__name__}: {exc}")
            try:
                counts.update(collect_schedule(args.days_ahead))
            except Exception as exc:  # noqa: BLE001
                ERRORS.append(f"schedule: {type(exc).__name__}: {exc}")
            counts.update(
                collect_boxscores(today - timedelta(days=args.lookback_days), today)
            )
    except Exception as exc:  # noqa: BLE001
        ERRORS.append(f"unexpected: {type(exc).__name__}: {exc}")
    finally:
        health = write_health(args.mode, counts)

    log("\n=== SUMMARY ===")
    log(json.dumps({k: health[k] for k in ("ok", "this_run", "data", "warning_count", "error_count")}, indent=2))
    for w in health["warnings"]:
        log(f"WARNING: {w}")
    for e in health["errors"]:
        log(f"ERROR: {e}")
    return 0 if health["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
