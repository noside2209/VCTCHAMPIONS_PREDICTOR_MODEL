# VCT Champions 2026 Predictor

An in-depth match predictor for the 16 teams at **VALORANT Champions 2026 (Shanghai, Sep 24 – Oct 18)**.
It also predicts any match between these teams **outside Champions**: pick any two teams, Bo1/Bo3/Bo5,
any map pool, neutral venue or home crowd.

For every match it produces:

- **Map veto**: who bans and picks what, and the decider
- **Map-by-map win probability** and the **predicted exact map score** (e.g. `PRX 13-10`), expected rounds and overtime chance
- **Series correct score**, with the full probability of every scoreline (2-0 / 2-1 / 1-2 / 0-2, or the Bo5 equivalents)
- **Head-to-head** records that fed into the prediction
- For the event: the **full predicted bracket** (every remaining group and playoff match with correct score), **Monte Carlo title odds**, and the **probability of every possible pairing**

No dependencies: pure Python 3.10+.

## Quick start

```bash
# Any match, any event (neutral venue, current 7-map pool by default)
python -m vct_predictor match PRX NRG --bo 5
python -m vct_predictor match "Team Liquid" KC --bo 3 --pool Ascent Haven Lotus Split Sunset
python -m vct_predictor match EDG T1 --bo 1 --home China      # Chinese crowd advantage
python -m vct_predictor match G2 PRX --no-h2h                 # ignore head-to-head history

# Teams
python -m vct_predictor rankings
python -m vct_predictor team "Paper Rex"

# Every pairing at once (120 matchups)
python -m vct_predictor matrix --bo 3

# Champions 2026 bracket forecast + title odds
python -m vct_predictor tournament --sims 20000

# Regenerate everything in reports/
python -m vct_predictor report

# Tests
python -m unittest discover -s tests -t .
```

Team names accept the full name, the tag (`PRX`, `100T`, `KC`, `NS`, `XLG`…), or an unambiguous fragment.
**Team A** (the first team you name) starts the veto.

## Generated reports (`reports/`)

| File | What's in it |
|---|---|
| `dashboard.html` | Interactive dashboard: match lab (all 240 ordered pairings × Bo1/Bo3/Bo5), predicted bracket, title odds, team sheets |
| `tournament_forecast.md` | Title odds, power rankings, predicted bracket with correct scores, possible opponents, full breakdown of each predicted match |
| `all_matchups_bo1.md` / `bo3` / `bo5` | Every one of the 120 matchups with veto, map scores and series score odds |
| `matchup_matrix_bo*.csv` | Row-team win % against column team |
| `map_win_probabilities.csv` | Map-level win % for every ordered pair on all 7 maps |
| `team_profiles.md` | Rosters, rating breakdown, map ratings, logged series |

## How the model works

1. **Team rating (Elo-style).**
   `rating = season prior + roster adjustment + form`
   - *Prior*: strength from the 2026 season: Masters Santiago (NS def. PRX 3-0), Masters London (Leviatán def. PRX 3-2; EDG 3rd, VIT 4th),
     EWC 2026 (100T def. NRG 3-1), and the Stage 2 regional finals.
   - *Roster*: built from each player's estimated impact (team average, plus a bonus for a true star).
   - *Form*: map-by-map Elo updates from every logged series in `data/history.json` and the Champions results so far, with
     recency decay (120-day half-life). Lopsided maps (13-2) count more than close ones (14-12).
2. **Map ratings.** Each team has an offset per map (comfort picks and permabans). Logged map results nudge these values.
   Abyss (just returned to the pool) and Summit (released June 2026) are shrunk toward 0 because there is little pro data on them.
3. **Head-to-head.** Recency-weighted net series wins between the two teams, worth ±10 Elo each and capped at ±25.
4. **Home crowd.** Optional +10 Elo for teams from the host region (China at Champions Shanghai).
5. **Map win probability** = logistic on the rating gap (400-point scale).
6. **Veto.** Official VCT formats (Bo3: ban-ban-pick-pick-ban-ban-decider; Bo5: ban-ban-pick-pick-pick-pick-decider;
   Bo1: six bans). Each team bans its worst remaining map and picks its best.
7. **Exact map scores.** A round-level model: first to 13, win-by-two overtime. The per-round win rate is solved so the model
   reproduces the map win probability. It varies from map to map (momentum and economy swings), which gives realistic blowout and
   overtime frequencies (~13% OT on even maps).
8. **Series score.** Exact distribution from the map probabilities in veto order.
9. **Tournament.** GSL groups, then an 8-team double-elimination playoff (Bo3; lower final and grand final Bo5). Real results are locked.
   The chalk bracket always advances the favourite with its most likely score. The Monte Carlo run (20,000 simulations) gives placement odds.

## Keeping it up to date

- **New Champions result:** add it to `data/champions_results.json` (match id such as `A-winners`, `UBQF2`, `GF`) with
  the series score and, optionally, map scores. Then run `python -m vct_predictor report`.
- **Results from other events:** add them to `data/history.json`. They update form, map ratings and head-to-head.
- **Roster change:** edit `players` in `data/teams.json`.
- **Playoff seeding:** the exact quarterfinal cross-over wasn't confirmed at data time. The default is A1-C2, B1-D2, C1-A2, D1-B2;
  change it in `data/event.json` → `playoff_upper_quarterfinals`.

## Data sources and caveats

Data as of **2026-09-30**. At that point the group stage openers were complete, and both PRX (Group C) and NRG (Group D)
had won their winners' matches to qualify for playoffs.

- Teams, groups, rosters, map pool, 2026 event results and Champions results were collected from public coverage
  (VLR.gg, Liquipedia, THESPIKE, Sheep Esports, esports.gg, Red Bull, GosuGamers).
- **Player impact numbers, season priors and map offsets are analyst estimates, not scraped stats.** Stat sites could not be
  reached from the build environment. They are calibrated to known 2026 results and map picks, and are meant to be edited.
- Some logged dates are approximate (e.g. Masters Santiago and China Stage 2 finals). This only affects recency weighting slightly.
- Esports is high-variance: treat outputs as probabilities, not certainties.
