"""Smoke test: which NBA data sources are reachable from THIS machine?

Run it on a GitHub Actions runner (the real test) and locally (for comparison).
Required checks fail the run loudly. Informational checks are logged only.
Standard library only, so it works before any dependencies are installed.
"""
import json
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

TIMEOUT = 20
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}
NBA_HEADERS = {"Referer": "https://www.nba.com/", "Origin": "https://www.nba.com"}

results = []


def fetch(url, headers=None):
    req = urllib.request.Request(url, headers={**HEADERS, **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.status, resp.read()


def record(name, url, required, ok, detail, seconds, data=None):
    entry = {
        "name": name,
        "url": url,
        "required": required,
        "ok": ok,
        "detail": detail,
        "seconds": seconds,
        "_data": data,
    }
    results.append(entry)
    tag = "PASS" if ok else ("FAIL" if required else "info-fail")
    print(f"[{tag}] {name}: {detail} ({seconds}s)")
    return entry


def check(name, url, required, validate=None, headers=None):
    start = time.time()
    data = None
    try:
        status, body = fetch(url, headers)
        detail = f"HTTP {status}, {len(body)} bytes"
        ok = status == 200
        if ok and validate:
            data = json.loads(body)
            detail += " | " + validate(data)
    except Exception as exc:  # noqa: BLE001 - we want every failure logged
        ok = False
        detail = f"{type(exc).__name__}: {exc}"
    return record(name, url, required, ok, detail, round(time.time() - start, 2), data)


def v_cdn_scoreboard(d):
    games = d["scoreboard"]["games"]
    return f"{len(games)} games on scoreboard dated {d['scoreboard'].get('gameDate')}"


def v_cdn_schedule(d):
    dates = d["leagueSchedule"]["gameDates"]
    return f"{len(dates)} game dates, season {d['leagueSchedule'].get('seasonYear')}"


def v_cdn_boxscore(d):
    g = d["game"]
    n = len(g["homeTeam"]["players"]) + len(g["awayTeam"]["players"])
    return f"game {g['gameId']} status={g.get('gameStatusText')}, {n} players"


def v_espn_injuries(d):
    teams = d.get("injuries", [])
    n = sum(len(t.get("injuries", [])) for t in teams)
    return f"{len(teams)} teams, {n} injury rows"


def v_espn_scoreboard(d):
    events = d.get("events", [])
    done = sum(1 for e in events if e["status"]["type"].get("completed"))
    return f"{len(events)} events, {done} completed"


def v_espn_box(d):
    teams = d["boxscore"]["players"]
    n = 0
    for t in teams:
        stats = t.get("statistics") or [{}]
        n += len(stats[0].get("athletes", []))
    return f"{len(teams)} teams, {n} player rows"


def check_espn_boxscore(now_et):
    """Find a finished game in the last 3 days and pull its box score."""
    base = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
    for back in (1, 2, 3):
        day = (now_et - timedelta(days=back)).strftime("%Y%m%d")
        sb = check(
            f"ESPN scoreboard {day}",
            f"{base}/scoreboard?dates={day}",
            required=False,
            validate=v_espn_scoreboard,
        )
        if not sb["ok"] or not sb["_data"]:
            continue
        done = [e for e in sb["_data"]["events"] if e["status"]["type"].get("completed")]
        if done:
            event_id = done[0]["id"]
            check(
                f"ESPN box score (event {event_id})",
                f"{base}/summary?event={event_id}",
                required=True,
                validate=v_espn_box,
            )
            return
    record("ESPN box score", base, True, False,
           "no finished game found in the last 3 days", 0)


def check_injury_pdf(now_et):
    """Official report PDFs. Try today and yesterday, newest slot first."""
    base = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
    slots = ["08_00PM", "07_00PM", "06_00PM", "05_00PM", "04_00PM", "03_00PM",
             "02_00PM", "01_00PM", "12_00PM", "11_00AM", "10_00AM", "09_00AM"]
    start = time.time()
    tried = 0
    for off in (0, 1):
        day = (now_et - timedelta(days=off)).strftime("%Y-%m-%d")
        for slot in slots:
            url = f"{base}{day}_{slot}.pdf"
            tried += 1
            try:
                status, body = fetch(url)
            except Exception:  # noqa: BLE001 - missing files return 403/404
                continue
            if status == 200:
                record("official injury report PDF", url, False, True,
                       f"HTTP 200, {len(body)} bytes (after {tried} tries)",
                       round(time.time() - start, 2))
                return
    record("official injury report PDF", base, False, False,
           f"no PDF found after {tried} guessed filenames",
           round(time.time() - start, 2))


def main():
    now_et = datetime.now(ZoneInfo("America/New_York"))
    print(f"Smoke test at {datetime.now(timezone.utc).isoformat()}")
    print(f"US Eastern now: {now_et.isoformat()}\n")

    # cdn.nba.com: informational now. ESPN is the required fallback.
    sb = check(
        "cdn.nba.com scoreboard",
        "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
        required=False,
        validate=v_cdn_scoreboard,
        headers=NBA_HEADERS,
    )
    game_id = None
    if sb["_data"]:
        games = sb["_data"]["scoreboard"]["games"]
        started = [g for g in games if g.get("gameStatus", 1) >= 2]
        pick = started[0] if started else (games[0] if games else None)
        game_id = pick["gameId"] if pick else None
    if game_id:
        check(
            f"cdn.nba.com boxscore ({game_id})",
            f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json",
            required=False,
            validate=v_cdn_boxscore,
            headers=NBA_HEADERS,
        )
    else:
        print("[skip] cdn boxscore: no scoreboard data")
    check(
        "cdn.nba.com full schedule",
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
        required=False,
        validate=v_cdn_schedule,
        headers=NBA_HEADERS,
    )

    # ESPN: required.
    check(
        "ESPN NBA injuries",
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries",
        required=True,
        validate=v_espn_injuries,
    )
    check(
        "ESPN NBA scoreboard",
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
        required=True,
        validate=v_espn_scoreboard,
    )
    check_espn_boxscore(now_et)

    check_injury_pdf(now_et)

    # Expected to fail from cloud runners. Informational only.
    check(
        "stats.nba.com (expected to fail on cloud runners)",
        "https://stats.nba.com/stats/commonallplayers?LeagueID=00&Season=2026-27&IsOnlyCurrentSeason=1",
        required=False,
        headers=NBA_HEADERS,
    )

    for r in results:
        r.pop("_data", None)
    with open("smoke_results.json", "w") as f:
        json.dump(
            {"run_at": datetime.now(timezone.utc).isoformat(), "results": results},
            f,
            indent=2,
        )

    failed = [r for r in results if r["required"] and not r["ok"]]
    print()
    if failed:
        print("REQUIRED CHECKS FAILED:", ", ".join(r["name"] for r in failed))
        sys.exit(1)
    print("All required checks passed.")


if __name__ == "__main__":
    main()
