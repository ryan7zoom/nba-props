"""Smoke test: which NBA data sources are reachable from THIS machine?

Run it on a GitHub Actions runner (the real test) and locally (for comparison).
Required checks fail the run loudly. Informational checks are logged only.
Standard library only, so it works before any dependencies are installed.
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

TIMEOUT = 20
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; nba-props-smoke-test)",
    "Accept": "application/json, */*",
}

results = []


def fetch(url, headers=None):
    req = urllib.request.Request(url, headers={**HEADERS, **(headers or {})})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.status, resp.read()


def check(name, url, required, validate=None, headers=None):
    start = time.time()
    entry = {"name": name, "url": url, "required": required}
    try:
        status, body = fetch(url, headers)
        detail = f"HTTP {status}, {len(body)} bytes"
        ok = status == 200
        data = None
        if ok and validate:
            data = json.loads(body)
            detail += " | " + validate(data)
        entry.update(ok=ok, detail=detail)
        entry["_data"] = data
    except Exception as exc:  # noqa: BLE001 - we want every failure logged
        entry.update(ok=False, detail=f"{type(exc).__name__}: {exc}")
    entry["seconds"] = round(time.time() - start, 2)
    results.append(entry)
    tag = "PASS" if entry["ok"] else ("FAIL" if required else "info-fail")
    print(f"[{tag}] {name}: {entry['detail']} ({entry['seconds']}s)")
    return entry


def v_cdn_scoreboard(d):
    games = d["scoreboard"]["games"]
    return f"{len(games)} games on scoreboard dated {d['scoreboard'].get('gameDate')}"


def v_espn_injuries(d):
    teams = d.get("injuries", [])
    n = sum(len(t.get("injuries", [])) for t in teams)
    return f"{len(teams)} teams, {n} injury rows"


def v_espn_scoreboard(d):
    return f"{len(d.get('events', []))} events"


def v_cdn_schedule(d):
    dates = d["leagueSchedule"]["gameDates"]
    return f"{len(dates)} game dates, season {d['leagueSchedule'].get('seasonYear')}"


def v_boxscore(d):
    g = d["game"]
    n = len(g["homeTeam"]["players"]) + len(g["awayTeam"]["players"])
    return f"game {g['gameId']} status={g.get('gameStatusText')}, {n} players"


def injury_pdf_urls():
    """Official report PDFs. URL pattern is a best guess from memory; verify."""
    now = datetime.now(timezone.utc) - timedelta(hours=5)  # rough US Eastern
    base = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
    urls = []
    for day_offset in (0, 1):
        day = (now - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        for hhmm in ("05_00PM", "01_00PM", "11_00AM"):
            urls.append(f"{base}{day}_{hhmm}.pdf")
    return urls


def main():
    print(f"Smoke test at {datetime.now(timezone.utc).isoformat()}\n")

    sb = check(
        "cdn.nba.com scoreboard",
        "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json",
        required=True,
        validate=v_cdn_scoreboard,
    )

    # Box score for one game, if the scoreboard has any
    game_id = None
    if sb.get("_data"):
        games = sb["_data"]["scoreboard"]["games"]
        started = [g for g in games if g.get("gameStatus", 1) >= 2]
        pick = started[0] if started else (games[0] if games else None)
        game_id = pick["gameId"] if pick else None
    if game_id:
        check(
            f"cdn.nba.com boxscore ({game_id})",
            f"https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json",
            required=False,  # upcoming games return nothing, so informational
            validate=v_boxscore,
        )
    else:
        print("[skip] boxscore: no games on today's scoreboard")

    check(
        "cdn.nba.com full schedule",
        "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json",
        required=False,
        validate=v_cdn_schedule,
    )
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

    # Official injury report PDF: try a few likely filenames, any hit counts.
    pdf_hit = False
    for url in injury_pdf_urls():
        e = check("official injury report PDF", url, required=False)
        if e["ok"]:
            pdf_hit = True
            break
    if not pdf_hit:
        print("[note] no official injury PDF matched the guessed filenames; "
              "the URL pattern or timing may be off, not necessarily the access.")

    # Expected to fail from cloud runners. Informational only.
    check(
        "stats.nba.com (expected to fail on cloud runners)",
        "https://stats.nba.com/stats/commonallplayers?LeagueID=00&Season=2026-27&IsOnlyCurrentSeason=1",
        required=False,
        headers={"Referer": "https://www.nba.com/", "Origin": "https://www.nba.com"},
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
