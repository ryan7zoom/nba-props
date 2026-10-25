"""Step 4a: official NBA injury report.

The NBA publishes an injury report PDF through the day (statuses: Out, Doubtful,
Questionable, Probable, Available). ESPN's feed only has Out / Day-To-Day, so
this adds the missing detail and, over time, a history we can use to measure how
often a player with each status actually plays.

Modes
  fetch      look for new report PDFs from the last --days days (default 2),
             parse them, keep a change log of statuses, and save the newest
             report as data/injuries_official_latest.csv.
  backfill   same, for every day from --start to --end (default today). Resumable:
             reports already processed are skipped.
  calibrate  join the status log with box scores and write data/play_probs.json:
             how often players with each status played.

Files (in data/)
  injuries_official_history.csv  one row each time a player's status or reason changed
  injuries_official_latest.csv   every row of the newest report
  injury_reports_seen.csv        reports already processed
  injury_report_health.json      what the last run found; errors fail the run
  play_probs.json                P(play | status), with the counts behind it

Needs pdfplumber (pip install pdfplumber).
"""
import argparse
import csv
import io
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

DATA = Path("data")
HISTORY = DATA / "injuries_official_history.csv"
LATEST = DATA / "injuries_official_latest.csv"
SEEN = DATA / "injury_reports_seen.csv"
HEALTH = DATA / "injury_report_health.json"
PLAY_PROBS = DATA / "play_probs.json"
PLAYER_GAMES = DATA / "player_games.csv"

ET = ZoneInfo("America/New_York")
BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
DEFAULT_NON_INJURY = 0.10  # "Questionable"/"Probable" next to a G League or two-way reason
DEFAULT_PLAY = {"Available": 0.97, "Probable": 0.90, "Questionable": 0.55, "Doubtful": 0.12, "Out": 0.02}
PRIOR_STRENGTH = 20  # defaults count as this many games of evidence

TEAMS = {  # ESPN code -> names as printed in the report
    "ATL": ["Atlanta Hawks"], "BOS": ["Boston Celtics"], "BKN": ["Brooklyn Nets"],
    "CHA": ["Charlotte Hornets"], "CHI": ["Chicago Bulls"], "CLE": ["Cleveland Cavaliers"],
    "DAL": ["Dallas Mavericks"], "DEN": ["Denver Nuggets"], "DET": ["Detroit Pistons"],
    "GS": ["Golden State Warriors"], "HOU": ["Houston Rockets"], "IND": ["Indiana Pacers"],
    "LAC": ["LA Clippers", "Los Angeles Clippers"], "LAL": ["Los Angeles Lakers", "LA Lakers"],
    "MEM": ["Memphis Grizzlies"], "MIA": ["Miami Heat"], "MIL": ["Milwaukee Bucks"],
    "MIN": ["Minnesota Timberwolves"], "NO": ["New Orleans Pelicans"], "NY": ["New York Knicks"],
    "OKC": ["Oklahoma City Thunder"], "ORL": ["Orlando Magic"], "PHI": ["Philadelphia 76ers"],
    "PHX": ["Phoenix Suns"], "POR": ["Portland Trail Blazers"], "SAC": ["Sacramento Kings"],
    "SA": ["San Antonio Spurs"], "TOR": ["Toronto Raptors"], "UTAH": ["Utah Jazz"],
    "WSH": ["Washington Wizards"],
}
TRI = {"GSW": "GS", "NOP": "NO", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

NORM_V = "2"  # bump when name or reason parsing changes; old history is then rebuilt automatically
HISTORY_FIELDS = ["report_et", "report_utc", "game_date_et", "game_time_et", "matchup", "team",
                  "player_raw", "player_norm", "status", "reason", "file", "norm_v"]
LATEST_FIELDS = HISTORY_FIELDS[:]

WARNINGS, ERRORS = [], []


def log(msg):
    print(msg, flush=True)


def dmy(iso):
    """2026-10-24 -> 24/10/2026 (all dates shown to people use this format)."""
    s = str(iso)
    return f"{s[8:10]}/{s[5:7]}/{s[0:4]}" if len(s) >= 10 and s[4] == "-" else s


def dmy_slot(name):
    """2026-10-24_05_30PM -> 24/10/2026 05:30PM"""
    return f"{dmy(name[:10])} {name[11:13]}:{name[14:]}" if len(name) >= 17 else name


def dmy_time(dt):
    return dt.strftime("%d/%m/%Y %H:%M")


NON_INJURY_RE = re.compile(r"g\s*league|two-?\s*way|not\s*with\s*team|suspension|personal|trade|coach", re.I)


def is_non_injury(reason):
    """G League assignments, two-way contracts, suspensions etc.: a different thing from a health status."""
    return bool(NON_INJURY_RE.search(reason or ""))


def warn(msg):
    WARNINGS.append(msg)
    log(f"WARNING: {msg}")


def err(msg):
    ERRORS.append(msg)
    log(f"ERROR: {msg}")


# ---------- names ----------

def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def norm_name(raw):
    """'Jones Jr., Derrick' and 'Derrick Jones Jr.' both become 'derrick jones'."""
    raw = raw.strip()
    if "," in raw:
        last, first = raw.split(",", 1)
        raw = f"{first.strip()} {last.strip()}"
    s = strip_accents(raw).lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = []
    for t in s.split():
        if t in SUFFIXES:
            continue
        for suf in ("iii", "jr", "sr", "iv", "ii"):  # suffix glued on, e.g. "porterjr"
            if t.endswith(suf) and len(t) >= len(suf) + 3:
                t = t[:-len(suf)]
                break
        toks.append(t)
    return " ".join(toks)


def last_first_key(norm):
    toks = norm.split()
    return (toks[-1], toks[0][:1]) if toks else ("", "")


# ---------- parsing the PDF text ----------

def _team_regex():
    alts = {}
    for code, names in TEAMS.items():
        for n in names:
            alts[re.sub(r"\s+", "", n).lower()] = code
    parts = []
    for n in sorted((n for names in TEAMS.values() for n in names), key=len, reverse=True):
        parts.append(r"\s*".join(re.escape(w) for w in n.split()))
    return re.compile(r"^(" + "|".join(parts) + r")\s*", re.I), alts


TEAM_RE, TEAM_LOOKUP = _team_regex()
DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})\s*")
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})\s*(AM|PM)?\s*\(?ET\)?\s*", re.I)
MATCHUP_RE = re.compile(r"^([A-Z]{2,4})@([A-Z]{2,4})\s*")
STATUS_RE = re.compile(r"(Out|Doubtful|Questionable|Probable|Available)")  # case-sensitive; may be glued to the name
SKIP_RE = re.compile(r"^(Injury\s*Report|Game\s*Date|Page\s*\d+\s*of\s*\d+)", re.I)
NOT_SUBMITTED_RE = re.compile(r"NOT\s*YET\s*SUBMITTED", re.I)


def to_24h(h, m, ampm):
    h = int(h)
    if ampm:
        ampm = ampm.upper()
        h = h % 12 + (12 if ampm == "PM" else 0)
    else:  # the report prints 12-hour times without AM/PM; NBA tip-offs are almost all evening
        if h <= 10:
            h += 12
        elif h == 11:
            pass
    return f"{h:02d}:{m}"


CATEGORY_RE = re.compile(r"^(Injury|Illness|G\s*League|Not\s*With|Personal|Suspension|Trade|Coach|Health|Return|Conditioning)", re.I)


def parse_text(text):
    """Turn the report's text into rows. Returns (rows, not_submitted_teams, unparsed_lines).

    Reason cells can wrap onto several lines, and the PDF centers them vertically, so the
    first part of a long reason can print on the line ABOVE the player's row. A loose line
    is therefore held back and attached to the next row if that row's own reason does not
    start a new reason, otherwise to the row before it.
    """
    rows, not_submitted, unparsed = [], [], []
    cur = {"date": "", "time": "", "matchup": "", "team": "", "team_raw": ""}
    last_row = None
    pending = []

    def flush_to_previous():
        nonlocal pending
        if pending:
            if last_row is not None:
                last_row["reason"] = (last_row["reason"] + " " + " ".join(pending)).strip()
            else:
                unparsed.extend(pending)
            pending = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or SKIP_RE.match(line):
            continue
        rest = line
        m = DATE_RE.match(rest)
        if m:
            cur["date"] = f"{m.group(3)}-{m.group(1)}-{m.group(2)}"
            rest = rest[m.end():]
            cur["time"] = ""
        m = TIME_RE.match(rest)
        if m:
            cur["time"] = to_24h(m.group(1), m.group(2), m.group(3))
            rest = rest[m.end():]
        m = MATCHUP_RE.match(rest)
        if m:
            a, b = TRI.get(m.group(1), m.group(1)), TRI.get(m.group(2), m.group(2))
            cur["matchup"] = f"{a}@{b}"
            rest = rest[m.end():]
        m = TEAM_RE.match(rest)
        if m:
            cur["team_raw"] = m.group(1)
            cur["team"] = TEAM_LOOKUP[re.sub(r"\s+", "", m.group(1)).lower()]
            rest = rest[m.end():]
        if NOT_SUBMITTED_RE.search(rest):
            flush_to_previous()
            not_submitted.append(cur["team"])
            last_row = None
            continue
        comma = rest.find(",")
        sm = STATUS_RE.search(rest, comma + 1) if comma >= 0 else None
        if sm:
            name = rest[:sm.start()].strip()
            own = rest[sm.end():].strip()
            if pending:
                if own and CATEGORY_RE.match(own):
                    flush_to_previous()          # loose lines belonged to the row above
                    reason = own
                else:
                    reason = " ".join(pending + ([own] if own else []))  # first part of this row's reason
                    pending = []
            else:
                reason = own
            last_row = {"game_date_et": cur["date"], "game_time_et": cur["time"], "matchup": cur["matchup"],
                        "team": cur["team"], "player_raw": name, "player_norm": norm_name(name),
                        "status": sm.group(1), "reason": reason}
            rows.append(last_row)
        elif len(rest) < 120 and not STATUS_RE.search(rest):
            pending.append(rest)
        else:
            unparsed.append(line)
    flush_to_previous()
    return rows, not_submitted, unparsed


def parse_pdf(data):
    import pdfplumber
    texts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            texts.append(page.extract_text() or "")
    text = "\n".join(texts)
    rows, ns, unparsed = parse_text(text)
    return rows, ns, unparsed, text


def check_rows(rows, unparsed, text, fname):
    """Fail loudly if the layout looks different from what we expect."""
    problems = []
    if not text.strip():
        problems.append("no text in PDF (scanned or empty?)")
    if not rows and re.search(r"Out|Questionable|Doubtful|Probable|Available", text):
        problems.append("status words present but no player rows parsed (layout changed?)")
    missing = [r for r in rows if not (r["game_date_et"] and r["team"] and r["matchup"])]
    if rows and len(missing) > 0.1 * len(rows):
        problems.append(f"{len(missing)}/{len(rows)} rows lack date/team/matchup")
    if rows and len(unparsed) > 0.25 * (len(rows) + len(unparsed)):
        problems.append(f"{len(unparsed)} unparsed lines vs {len(rows)} rows")
    if problems:
        sample = "\n".join(text.splitlines()[:25])
        return f"{dmy_slot(fname)}: " + "; ".join(problems) + f". First lines of the PDF text:\n{sample}"
    return None


# ---------- finding and downloading reports ----------

def slot_list(day):
    out = []
    for hh in range(1, 13):
        for mm in (0, 15, 30, 45):
            for ap in ("AM", "PM"):
                name = f"{day:%Y-%m-%d}_{hh:02d}_{mm:02d}{ap}"
                h24 = hh % 12 + (12 if ap == "PM" else 0)
                out.append((name, datetime(day.year, day.month, day.day, h24, mm, tzinfo=ET)))
    return sorted(out, key=lambda x: x[1])


def fetch_pdf(url, retries=2):
    """Returns PDF bytes, None if the file does not exist, raises on network trouble."""
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (nba-props)"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            if data[:4] != b"%PDF":
                return None
            return data
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):  # missing keys on this host often come back as 403
                return None
            last = e
        except Exception as e:  # timeouts, resets
            last = e
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def probe_day(day, seen, now_et):
    slots = [(n, t) for n, t in slot_list(day) if t <= now_et and n not in seen]
    found, errors = [], 0

    def one(item):
        name, t = item
        try:
            return name, t, fetch_pdf(BASE + name + ".pdf")
        except Exception:
            return name, t, "ERR"

    with ThreadPoolExecutor(max_workers=12) as ex:
        for name, t, data in ex.map(one, slots):
            if data == "ERR":
                errors += 1
            elif data:
                found.append((name, t, data))
    return found, errors, len(slots)


# ---------- storage ----------

def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path, fields, rows):
    DATA.mkdir(exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerows(rows)


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def last_known(history_rows):
    """(game_date, team, player_norm) -> (status, reason) from the change log."""
    known = {}
    for r in sorted(history_rows, key=lambda r: r["report_utc"]):
        known[(r["game_date_et"], r["team"], r["player_norm"])] = (r["status"], r["reason"])
    return known


def process_reports(found, known):
    """Chronological: add change-log rows. Returns (new history rows, latest report rows, stats)."""
    new_hist, latest, latest_t = [], [], None
    stats = Counter()
    ok_names = []
    for name, t, data in sorted(found, key=lambda x: x[1]):
        try:
            rows, ns, unparsed, text = parse_pdf(data)
        except Exception as e:
            err(f"{dmy_slot(name)}: could not read PDF ({type(e).__name__}: {e})")
            stats["failed"] += 1
            continue
        problem = check_rows(rows, unparsed, text, name)
        if problem:
            err(problem)
            stats["failed"] += 1
            continue
        if unparsed:
            warn(f"{dmy_slot(name)}: {len(unparsed)} unparsed lines, e.g. {unparsed[0]!r}")
        stats["parsed"] += 1
        stats["rows"] += len(rows)
        ok_names.append((name, t))
        report_utc = t.astimezone(timezone.utc).isoformat(timespec="minutes")
        base = {"report_et": t.isoformat(timespec="minutes"), "report_utc": report_utc, "file": name,
                "norm_v": NORM_V}
        teams_in_report = set()
        present = set()
        for r in rows:
            key = (r["game_date_et"], r["team"], r["player_norm"])
            teams_in_report.add((r["game_date_et"], r["team"]))
            present.add(key)
            if known.get(key) != (r["status"], r["reason"]):
                new_hist.append({**base, **r})
                known[key] = (r["status"], r["reason"])
        for key, (st, rs) in list(known.items()):  # listed earlier, gone now = cleared
            if (key[0], key[1]) in teams_in_report and key not in present and st != "Removed":
                new_hist.append({**base, "game_date_et": key[0], "game_time_et": "", "matchup": "",
                                 "team": key[1], "player_raw": "", "player_norm": key[2],
                                 "status": "Removed", "reason": ""})
                known[key] = ("Removed", "")
        if latest_t is None or t >= latest_t:
            latest_t = t
            latest = [{**base, **r} for r in rows]
        stats["not_submitted_team_lines"] += len(ns)
    return new_hist, latest, stats, ok_names


# ---------- fetch / backfill ----------

def run_fetch(start_day, end_day, mode):
    DATA.mkdir(exist_ok=True)
    now_et = datetime.now(timezone.utc).astimezone(ET)
    old = read_csv(HISTORY)
    if old and old[0].get("norm_v") != NORM_V:
        log("Old injury history was built with an older name/reason parser: rebuilding it from the reports.")
        for path in (HISTORY, SEEN, LATEST):
            if path.exists():
                path.unlink()
    seen_rows = read_csv(SEEN)
    seen = {r["file"] for r in seen_rows}
    hist = read_csv(HISTORY)
    known = last_known(hist)
    total_found = total_err = total_probed = 0
    day = start_day
    while day <= end_day:
        found, errors, probed = probe_day(day, seen, now_et)
        total_found += len(found)
        total_err += errors
        total_probed += probed
        if found:
            new_hist, latest, stats, ok = process_reports(found, known)
            if new_hist:
                append_csv(HISTORY, HISTORY_FIELDS, [{k: r.get(k, "") for k in HISTORY_FIELDS} for r in new_hist])
            if latest and mode == "fetch":
                write_csv(LATEST, LATEST_FIELDS, [{k: r.get(k, "") for k in LATEST_FIELDS} for r in latest])
            # mark as seen only the reports that were handled without a parse error
            append_csv(SEEN, ["file", "report_utc", "rows"],
                       [{"file": n, "report_utc": t.astimezone(timezone.utc).isoformat(timespec="minutes"),
                         "rows": ""} for n, t in ok])
            seen.update(n for n, _ in ok)
            log(f"{dmy(day)}: {len(found)} new reports, {stats['rows']} rows, {len(new_hist)} status changes")
        elif probed:
            log(f"{dmy(day)}: no new reports ({probed} slots probed)")
        day += timedelta(days=1)

    # latest file for backfill runs: newest report seen overall is rebuilt from history is not needed
    health = {
        "run_at": dmy_time(datetime.now(timezone.utc)) + " UTC", "mode": mode,
        "ok": not ERRORS, "days": f"{dmy(start_day.isoformat())} to {dmy(end_day.isoformat())}", "slots_probed": total_probed,
        "reports_found": total_found, "probe_errors": total_err,
        "warnings": WARNINGS[:50], "errors": ERRORS[:20],
    }
    if mode == "fetch":
        latest_rows = read_csv(LATEST)
        health["latest_report_et"] = (dmy_time(datetime.fromisoformat(latest_rows[0]["report_et"])) + " ET") if latest_rows else None
        health["latest_rows"] = len(latest_rows)
        if not latest_rows:
            warn("no official report saved yet")
        else:
            age = now_et - datetime.fromisoformat(latest_rows[0]["report_et"])
            health["latest_report_age_hours"] = round(age.total_seconds() / 3600, 1)
            if age > timedelta(hours=36):
                warn(f"newest official report is {age.total_seconds() / 3600:.0f} hours old")
    if total_probed and total_err > 0.3 * total_probed:
        err(f"{total_err}/{total_probed} probes hit network errors")
    if mode == "fetch" and total_found == 0 and not read_csv(SEEN):
        err("no injury report PDF found in the last days and none stored: the URL pattern may have changed")
    health["ok"] = not ERRORS
    health["warnings"], health["errors"] = WARNINGS[:50], ERRORS[:20]
    HEALTH.write_text(json.dumps(health, indent=2), encoding="utf-8")
    log(f"Found {total_found} reports. Warnings: {len(WARNINGS)}, errors: {len(ERRORS)}")
    return 1 if ERRORS else 0


# ---------- calibration ----------

LEAD_MINUTES = 120  # judge statuses as they stood this long before tip-off (a realistic time to bet)


def status_at_lead(hist_rows, lead_minutes=LEAD_MINUTES):
    """(game_date, team, player_norm) -> (status, reason) as of `lead_minutes` before tip-off."""
    by_key = defaultdict(list)
    for r in hist_rows:
        by_key[(r["game_date_et"], r["team"], r["player_norm"])].append(r)
    out = {}
    for key, rs in by_key.items():
        rs.sort(key=lambda r: r["report_utc"])
        tip_str = next((r["game_time_et"] for r in rs if r["game_time_et"]), "")
        if not tip_str:
            continue
        h, m = tip_str.split(":")
        d = datetime.fromisoformat(key[0])
        cutoff = (datetime(d.year, d.month, d.day, int(h), int(m), tzinfo=ET)
                  - timedelta(minutes=lead_minutes)).astimezone(timezone.utc).isoformat(timespec="minutes")
        usable = [r for r in rs if r["report_utc"] <= cutoff]
        if usable:
            last = usable[-1]
            st = "Available" if last["status"] == "Removed" else last["status"]
            reason = next((r["reason"] for r in reversed(usable) if r["reason"]), "")
            out[key] = (st, reason)
    return out


def run_calibrate():
    hist = read_csv(HISTORY)
    now_txt = dmy_time(datetime.now(timezone.utc)) + " UTC"
    if not hist:
        log("No official injury history yet; keeping default play probabilities.")
        PLAY_PROBS.write_text(json.dumps({"source": "defaults", "probs": DEFAULT_PLAY,
                                          "non_injury": DEFAULT_NON_INJURY, "stats": {}, "as_of": now_txt},
                                         indent=2), encoding="utf-8")
        return 0
    played, team_days, by_team_last = set(), set(), defaultdict(set)
    with open(PLAYER_GAMES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["season_type"] not in ("2", "3", "5"):
                continue
            team_days.add((r["game_date_et"], r["team"]))
            if r["dnp"] != "1" and float(r["min"] or 0) > 0:
                n = norm_name(r["player"])
                played.add((r["game_date_et"], r["team"], n))
                by_team_last[(r["game_date_et"], r["team"])].add((last_first_key(n), n))
    snap = status_at_lead(hist)
    counts = defaultdict(lambda: [0, 0])  # status -> [n, played], health reasons only
    non_inj = [0, 0]
    reason_counts = defaultdict(lambda: [0, 0])
    skipped_no_box = 0
    for key, (st, reason) in snap.items():
        d, team, n = key
        if (d, team) not in team_days:
            skipped_no_box += 1
            continue
        did = key in played
        if not did:  # fall back to last-name + first-initial if unique on that team
            lf = last_first_key(n)
            cands = [full for k, full in by_team_last[(d, team)] if k == lf]
            did = len(cands) == 1 and (d, team, cands[0]) in played
        if is_non_injury(reason) and st != "Available":
            non_inj[0] += 1
            non_inj[1] += int(did)
            continue
        counts[st][0] += 1
        counts[st][1] += int(did)
        if st == "Questionable":
            low = reason.lower()
            tag = "rest/management" if ("rest" in low or "management" in low) else (
                "illness" if "illness" in low else "injury")
            reason_counts[tag][0] += 1
            reason_counts[tag][1] += int(did)
    probs, stats = {}, {}
    for st in STATUSES:
        n, p = counts.get(st, [0, 0])
        prior = DEFAULT_PLAY[st]
        probs[st] = round((p + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH), 4)
        stats[st] = {"n": n, "played": p, "raw_rate": round(p / n, 3) if n else None, "default": prior}
    p_non = round((non_inj[1] + PRIOR_STRENGTH * DEFAULT_NON_INJURY) / (non_inj[0] + PRIOR_STRENGTH), 4)
    out = {"source": "history", "status_taken_minutes_before_tip": LEAD_MINUTES, "probs": probs,
           "non_injury": p_non,
           "non_injury_counts": {"n": non_inj[0], "played": non_inj[1]}, "stats": stats,
           "questionable_by_reason": {k: {"n": v[0], "played": v[1], "raw_rate": round(v[1] / v[0], 3)}
                                      for k, v in reason_counts.items() if v[0]},
           "players_with_status_but_no_box_score_day": skipped_no_box, "as_of": now_txt}
    PLAY_PROBS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"Probability a player plays, by official status {LEAD_MINUTES} minutes before tip-off "
        f"(history vs default):")
    for st in STATUSES:
        s = stats[st]
        log(f"  {st:<13} n={s['n']:<6} played={s['played']:<6} raw={s['raw_rate']}  used={probs[st]}")
    log(f"  G League / two-way / suspension etc. (not health): n={non_inj[0]} played={non_inj[1]} used={p_non}")
    for k, v in out["questionable_by_reason"].items():
        log(f"  Questionable ({k}): n={v['n']} raw={v['raw_rate']}")
    return 0


def parse_user_date(text):
    """Accepts dd/mm/yyyy (preferred) or yyyy-mm-dd."""
    text = text.strip()
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        d, m, y = text.split("/")
        return date(int(y), int(m), int(d))
    return date.fromisoformat(text)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fetch", "backfill", "calibrate"], required=True)
    p.add_argument("--days", type=int, default=2)
    p.add_argument("--start", help="dd/mm/yyyy")
    p.add_argument("--end", help="dd/mm/yyyy")
    args = p.parse_args(argv)
    today = datetime.now(timezone.utc).astimezone(ET).date()
    if args.mode == "calibrate":
        return run_calibrate()
    if args.mode == "fetch":
        return run_fetch(today - timedelta(days=args.days), today, "fetch")
    if not args.start:
        raise SystemExit("--start dd/mm/yyyy is required for backfill")
    end = parse_user_date(args.end) if args.end else today
    return run_fetch(parse_user_date(args.start), end, "backfill")


if __name__ == "__main__":
    sys.exit(main())
