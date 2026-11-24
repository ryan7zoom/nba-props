# NBA props

Player prop distributions, a game model, and a daily card of priced bets, built from ESPN box scores and
DraftKings/FanDuel lines. Everything is walk-forward tested; nothing is tuned on the games it is scored on.

## What it does

For every player in tonight's games it produces a full distribution of points, rebounds, assists and their
combinations, a calibrated chance of any line, and the price that line is worth taking at. Two pages read the
output: `index.html` (games, builders, team totals) and `shortlist.html` (the card and the safety ladder).

## Pipeline

Run from the repository root, in this order. The GitHub workflows do this for you.

| Step | Script | Writes |
|---|---|---|
| 1. Data | `update_data.py --mode daily` | `player_games.csv`, `schedule.csv`, `injuries_latest.csv` |
| 2. Official report | `injury_report.py --mode fetch` | `injuries_official_latest.csv`, `play_probs.json` |
| 3. Appearance + calibration | `calibrate.py` | `appear_rates.json`, `calibration.json` |
| 4. Rates | `baseline_rates.py --mode estimate` | `player_rates.csv` |
| 5. Minutes | `minutes_model.py --mode project` | `minutes_projection.csv` |
| 6. Defense | `defense_effects.py --mode estimate` | `defense_effects.csv` |
| 7. Distributions | `simulate_props.py --mode project` | `projections.csv`, `projections_latest.json` |
| 8. Lines | `lines.py` | `lines_latest.json`, `lines_history.csv` |
| 9. Grade + refit | `grade_lines.py`, `calibrate.py` | `lines_grade.json`, `calibration.json` |
| 10. Safety ladder | `alt_lines.py` | `alt_lines_latest.json`, `alt_lines.csv` |
| 11. Team totals | `team_totals.py` | `team_totals_latest.json`, `game_lines_history.csv` |
| 12. Game model | `game_model.py project` | `games_latest.json` |
| 13. Card | `shortlist.py` | `shortlist_latest.json`, `picks_log.csv` |
| 14. Record | `shortlist.py --grade` | `shortlist_grade.json` |

Backtests (slow, run them when you change a model, not daily):

```
python scripts/baseline_rates.py --mode backtest
python scripts/minutes_model.py --mode backtest
python scripts/defense_effects.py --mode both
python scripts/simulate_props.py --mode backtest
python scripts/game_model.py backtest
python scripts/team_totals.py --grade          # needs a few weeks of recorded game lines
```

## How to use the safety ladder

`data/alt_lines.csv` has one row per player, market, side and confidence level:

| column | meaning |
|---|---|
| `line` | the **tightest** line that still clears `level` for that side, i.e. the best price available at that safety |
| `our_p` | our chance of that side, after the calibration measured on real closing lines |
| `fair_price` | 1 / `our_p` — the break-even decimal price |
| `min_price` | the smallest decimal price worth taking after the edge you require (default 2%) |
| `book_price` | the price the feed has at that exact line, when it has one |
| `edge` | `our_p` × the price − 1, at the book's price |

So to shop a bookmaker's alternate lines: find the player, pick your confidence (0.90 is a reasonable default),
and take the line shown **if** the price on offer is at or above `min_price`. If the book's price is listed and
`worth_taking` is true, that is a live edge; if no price is listed, the number is the model's read only and you
are comparing it against whatever your bookie offers.

`shortlist.html` renders the same ladder as a table with a search box and level/market switches.

## The two corrections that make the numbers honest

**Calibration (`calibration.json`).** The simulation's probability is over-confident outside the middle: on
4,093 real closing lines the model said 36% over when 46% happened, and 63% when 51% happened. `calibrate.py`
fits a monotone correction on the observed data and one blend weight against the bookmaker's de-vigged price, and
reports whether each is worth using on a held-out third of the days. Without a price for a line (an alternate
your bookie has not quoted) the correction stands alone.

**Appearance priors (`appear_rates.json`).** "Not on the injury report" was treated as a 100% chance of playing.
Measured over three seasons it is 88% for a 20+ minute player, 81% for 14-20, 64% for 8-14 and 42% below that.
The number is reported as `p_appear` next to `p_play` (which stays the news-based number the other steps use), so
nothing downstream changed behaviour silently.

## Workflows

| Workflow | When | Does |
|---|---|---|
| `daily.yml` | 07:20, 12:20 UTC | everything, including the backtest-free rebuild and the card |
| `pregame.yml` | 21:05, 23:20, 00:20 UTC (~2h before each slate) | fresh injuries + fresh prices, re-project, rebuild the card |
| `lines.yml`, `minutes.yml`, `rates.yml`, `simulate.yml`, `defense.yml` | manual | one step, with its backtest |
| `team_backtest.yml`, `zones_fetch.yml`, `smoke-test.yml` | manual / daily | model and data-source checks |

Scheduled runs never lock picks; a manual `Run workflow` locks the next slate into `picks_log.csv`. Picks also
freeze automatically once their game tips off.

## Data notes

- `data/` is committed, because the project is stateless between runs. `player_games.csv` and `shot_zones.csv`
  are the big ones.
- `shot_zones.csv` is collected but no model reads it yet. It is the best available proxy for shot quality; the
  obvious next step is modelling points as attempts × conversion by zone instead of one overdispersed count.
- The market total/spread is recorded every run in `game_lines_history.csv`; `team_totals.py --grade` becomes
  meaningful after a couple of hundred games.

## Tests

```
pip install -r requirements.txt
python -m pytest tests -q
```

## Honest limits

- The model does not beat the closing line as a point forecast (MAE 5.14 vs the book's 5.05 on points). The edge
  in the card comes from the calibrated gap between our number and the price on offer, not from out-predicting
  the market.
- 60 bets is noise. The record needs 250+ before it means anything, and `shortlist_grade.json` says so.
- Team totals are recorded but not yet calibrated; their ladder is the ratings model's distribution, not a
  market-calibrated one.
