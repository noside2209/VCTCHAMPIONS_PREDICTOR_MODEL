# VCT Champions 2026 Predictor

An in-depth match predictor for the 16 teams at **VALORANT Champions 2026 (Shanghai, Sep 24 – Oct 18)**.
It also predicts any match between these teams **outside Champions**: pick any two teams, Bo1/Bo3/Bo5,
any map pool, neutral venue or home crowd.

For every match it produces:

- **Map veto**: who bans and picks what, and the decider
- **Map-by-map win probability** and the **predicted exact map score** (e.g. `PRX 13-10`), expected rounds and overtime chance
- **Series correct score**, with the full probability of every scoreline (2-0 / 2-1 / 1-2 / 0-2, or the Bo5 equivalents)
- **Head-to-head** records that fed into the prediction
- **Every current player**: role, IGL, agent pool, the agent each player is projected to play on each map, and a
  **projected box score** for every map and the whole series (K / D / A, K/D, ACS, ADR, KAST, first kills, first deaths,
  rating), plus a projected MVP and top fragger
- For the event: the **full predicted bracket** (every remaining group and playoff match with correct score), **Monte Carlo title odds**, and the **probability of every possible pairing**

No dependencies: pure Python 3.10+.

## Quick start

```bash
# Any match, any event (neutral venue, current 7-map pool by default)
python -m vct_predictor match PRX NRG --bo 5
python -m vct_predictor match "Team Liquid" KC --bo 3 --pool Ascent Haven Lotus Split Sunset
python -m vct_predictor match EDG T1 --bo 1 --home China      # Chinese crowd advantage
python -m vct_predictor match G2 PRX --no-h2h                 # ignore head-to-head history
python -m vct_predictor match VIT FUT --players series        # series-level player stats only (or: none)

# Teams
python -m vct_predictor rankings
python -m vct_predictor team "Paper Rex"      # roster, roles, agent pools, agents by map, map ratings

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

## Simulate: random play-outs

The prediction shows the single most likely result. To see the range of what can happen, simulate:

```bash
python -m vct_predictor simulate PRX NRG --bo 5              # one random play-out with box scores
python -m vct_predictor simulate PRX NRG --bo 3 --runs 10    # ten runs + a tally of the results
python -m vct_predictor simulate PRX NRG --seed 42           # same seed = same run, for sharing
python -m vct_predictor simulate-tournament --runs 3         # the rest of Champions played out at random
```

Each run samples the veto, since teams don't always make the textbook call when two maps are close. It plays every map
round by round, and each map gets its own form swing, which is where blowouts and comebacks come from. It then draws
each player's whole-number box score from their role and impact, with random form. Upsets, 13-2s and double-overtime
maps turn up about as often as the model thinks they should.

The dashboard has the same engine behind the **Simulate match** / **Run 100** buttons (match lab) and
**Simulate tournament** / **Run 100** (bracket tab). Tallies show how the runs compare to the model's odds.

## Generated reports (`reports/`)

| File | What's in it |
|---|---|
| `dashboard.html` | Interactive dashboard: match lab (all 240 ordered pairings × Bo1/Bo3/Bo5, with player box scores per map), predicted bracket, title odds, team sheets |
| `tournament_forecast.md` | Title odds, power rankings, predicted bracket with correct scores, possible opponents, full breakdown of each predicted match including per-map player box scores |
| `all_matchups_bo1.md` / `bo3` / `bo5` | Every one of the 120 matchups with veto, map scores, series score odds and series-level player projections |
| `matchup_matrix_bo*.csv` | Row-team win % against column team |
| `map_win_probabilities.csv` | Map-level win % for every ordered pair on all 7 maps |
| `team_profiles.md` | Rosters with roles, IGLs, agent pools, projected agents on every map, rating breakdown, map ratings, logged series |

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
   reproduces the map win probability. It swings from map to map (sd 0.11, for momentum, economy and form), which gives
   realistic frequencies: roughly 25–35% of maps are blowouts (loser on 5 or fewer rounds) and ~10% go to overtime.
8. **Series score.** Exact distribution from the map probabilities in veto order.
9. **Players.** `data/players.json` holds each player's role, IGL flag and agent pool, plus a standard composition for every map.
   Agents are assigned per map by matching the comp to each player's pool (flex players take the role of the agent they are on).
   Box scores come from the expected rounds. Team kills per round rise with the share of rounds won, and one team's kills equal the
   other's deaths, so box scores always balance. Kills, deaths, assists and first kills/deaths are shared out by role × player impact.
   ACS, ADR, KAST and rating are derived from those per-round rates. The series projection weights each map by the chance it is played.
   The MVP is the highest-rated player on the predicted winner.
10. **Tournament.** GSL groups, then an 8-team double-elimination playoff (Bo3; lower final and grand final Bo5). Real results are locked.
   The chalk bracket always advances the favourite with its most likely score. The Monte Carlo run (20,000 simulations) gives placement odds.

## Syncing real player data from VLR.gg

Run this on your own computer (it needs internet access to vlr.gg):

```bash
python -m vct_predictor sync-vlr                 # all 80 players, last 90 days (~3-4 minutes)
python -m vct_predictor sync-vlr --team LOUD     # just one team
python -m vct_predictor sync-vlr --timespan all  # career numbers instead of last 90 days
python -m vct_predictor report                   # rebuild the reports + dashboard with the new data
```

For each player it reads the agent table on their VLR profile and updates `data/players.json` with:
- the agent pool, ordered by rounds played (agents under 5% of rounds are dropped)
- the role those agents imply
- the VLR id
- round-weighted stats (rating, ACS, ADR, KAST, KPR, DPR, APR, FKPR, FDPR)

It also sets each player's impact in `data/teams.json` to their VLR rating (skip this with `--keep-impact`).
Once synced, projected box scores use each player's real per-round rates instead of role estimates.

- Pools marked `user-verified` (e.g. tkzin: Neon, Waylay) are kept unless you pass `--overwrite-verified`.
- If a player can't be matched (common names), the command says so. Add `"vlr_id": <number from the VLR profile URL>`
  to that player in `data/players.json` and run it again.
- Agent assignment never puts a player on an agent outside his pool when he has one of the same role. A Neon/Waylay
  duelist plays Neon even on maps where the meta comp lists Jett.

## Keeping it up to date

- **New Champions result:** add it to `data/champions_results.json` (match id such as `A-winners`, `UBQF2`, `GF`) with
  the series score and, optionally, map scores. Then run `python -m vct_predictor report`.
- **Results from other events:** add them to `data/history.json`. They update form, map ratings and head-to-head.
- **Roster change:** edit `players` in `data/teams.json` (impact) and the matching entry in `data/players.json` (role, agents).
- **Meta shift:** edit `map_comps` in `data/players.json`.
- **Playoff seeding:** the exact quarterfinal cross-over wasn't confirmed at data time. The default is A1-C2, B1-D2, C1-A2, D1-B2;
  change it in `data/event.json` → `playoff_upper_quarterfinals`.

## Data sources and caveats

Data as of **2026-09-30**. At that point the group stage openers were complete, and both PRX (Group C) and NRG (Group D)
had won their winners' matches to qualify for playoffs.

- Teams, groups, rosters, map pool, 2026 event results and Champions results were collected from public coverage
  (VLR.gg, Liquipedia, THESPIKE, Sheep Esports, esports.gg, Red Bull, GosuGamers).
- **Player impact numbers, season priors, map offsets, agent pools and projected player stats are model estimates, not scraped stats.**
  Roles marked `reported` come from 2026 roster coverage, `user-verified` pools were corrected by hand, and `estimated`
  roles are best guesses. Run `sync-vlr` to replace them with real VLR data. Stat sites could not be
  reached from the build environment. They are calibrated to known 2026 results and map picks, and are meant to be edited.
- Some logged dates are approximate (e.g. Masters Santiago and China Stage 2 finals). This only affects recency weighting slightly.
- Esports is high-variance: treat outputs as probabilities, not certainties.
