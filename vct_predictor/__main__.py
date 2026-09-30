"""Command line interface.

Examples:
    python -m vct_predictor match PRX NRG --bo 5
    python -m vct_predictor match "Team Liquid" "Karmine Corp" --bo 3 --pool Ascent Haven Lotus Split Sunset
    python -m vct_predictor match EDG T1 --home China
    python -m vct_predictor rankings
    python -m vct_predictor team "Paper Rex"
    python -m vct_predictor tournament --sims 20000
    python -m vct_predictor matrix --bo 3
    python -m vct_predictor report
    python -m vct_predictor sync-vlr            # refresh agent pools + stats from VLR.gg
    python -m vct_predictor simulate PRX NRG --bo 5 --runs 5   # random play-outs, not just the favourite
    python -m vct_predictor simulate-tournament --runs 3
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from .data import load_dataset
from .players import PlayerModel
from .ratings import RatingModel
from .report import (format_series, format_sim_series, pct, power_rankings_md, team_profiles_md,
                     tournament_md, write_reports)
from .series import predict_series
from .tournament import Tournament

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vct_predictor", description="VCT match & Champions 2026 predictor")
    ap.add_argument("--data", default=None, help="data directory (default: ./data)")
    ap.add_argument("--as-of", default=None, help="date for recency weighting, YYYY-MM-DD")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("match", help="predict any match between two modelled teams (any event)")
    m.add_argument("team_a", help="team starting the veto (name or tag)")
    m.add_argument("team_b")
    m.add_argument("--bo", type=int, default=3, choices=(1, 3, 5))
    m.add_argument("--pool", nargs="+", default=None, help="custom map pool (default: current 7-map pool)")
    m.add_argument("--home", default=None, help="region with crowd advantage, e.g. China / EMEA (default: neutral)")
    m.add_argument("--no-h2h", action="store_true", help="ignore head-to-head adjustment")
    m.add_argument("--players", choices=("full", "series", "none"), default="full",
                   help="player projections: per-map box scores + series (full), series only, or none")

    sm = sub.add_parser("simulate", help="play a match out at random, round by round (different every run)")
    sm.add_argument("team_a")
    sm.add_argument("team_b")
    sm.add_argument("--bo", type=int, default=3, choices=(1, 3, 5))
    sm.add_argument("--runs", type=int, default=1, help="number of play-outs to show")
    sm.add_argument("--seed", type=int, default=None, help="repeat the exact same runs")
    sm.add_argument("--pool", nargs="+", default=None)
    sm.add_argument("--home", default=None)
    sm.add_argument("--no-box", action="store_true", help="hide player box scores")

    st = sub.add_parser("simulate-tournament", help="play the rest of Champions 2026 out at random")
    st.add_argument("--runs", type=int, default=1)
    st.add_argument("--seed", type=int, default=None)

    sub.add_parser("rankings", help="power rankings")
    t = sub.add_parser("team", help="team profile")
    t.add_argument("name")

    tr = sub.add_parser("tournament", help="Champions 2026 bracket forecast")
    tr.add_argument("--sims", type=int, default=20000)

    mx = sub.add_parser("matrix", help="win%% for every pairing")
    mx.add_argument("--bo", type=int, default=3, choices=(1, 3, 5))
    mx.add_argument("--home", default=None)

    sv = sub.add_parser("sync-vlr", help="pull agent pools + stats from VLR.gg player pages (needs internet)")
    sv.add_argument("--team", default=None, help="only this team (name or tag)")
    sv.add_argument("--timespan", default="90d", choices=("30d", "60d", "90d", "all"))
    sv.add_argument("--keep-impact", action="store_true", help="don't overwrite impact with the VLR rating")
    sv.add_argument("--overwrite-verified", action="store_true", help="also overwrite user-verified agent pools")

    rp = sub.add_parser("report", help="write all reports to ./reports")
    rp.add_argument("--out", default=str(ROOT / "reports"))
    rp.add_argument("--sims", type=int, default=20000)

    args = ap.parse_args(argv)
    ds = load_dataset(args.data) if args.data else load_dataset()
    model = RatingModel(ds, args.as_of)

    if args.cmd == "match":
        a, b = ds.find_team(args.team_a).name, ds.find_team(args.team_b).name
        pool = args.pool
        if pool:
            known = {x.lower(): x for x in model.pool}
            pool = [known.get(p.lower(), p.title()) for p in pool]
        pred = predict_series(model, a, b, args.bo, pool, args.home, not args.no_h2h)
        print(format_series(pred, model, players=PlayerModel(ds), player_detail=args.players))
    elif args.cmd == "simulate":
        import random
        from collections import Counter
        from .simulate import simulate_series
        a, b = ds.find_team(args.team_a).name, ds.find_team(args.team_b).name
        pool = args.pool
        if pool:
            known = {x.lower(): x for x in model.pool}
            pool = [known.get(p.lower(), p.title()) for p in pool]
        rng = random.Random(args.seed)
        pm = PlayerModel(ds)
        tally = Counter()
        for i in range(1, args.runs + 1):
            sim = simulate_series(model, a, b, args.bo, pool, args.home, rng, pm)
            sw = sim.score if sim.winner == a else sim.score[::-1]
            tally[f"{sim.winner} {sw[0]}-{sw[1]}"] += 1
            print(format_sim_series(sim, model, f"Run {i}: {a} vs {b}", box_scores=not args.no_box))
        if args.runs > 1:
            print("Results across runs: " + ", ".join(f"{k} x{v}" for k, v in tally.most_common()))
    elif args.cmd == "simulate-tournament":
        import random
        from collections import Counter
        from .simulate import simulate_tournament
        from .tournament import stage_label
        rng = random.Random(args.seed)
        tour = Tournament(ds, model)
        champs = Counter()
        tag = lambda n: model.ratings[n].team.tag  # noqa: E731
        for i in range(1, args.runs + 1):
            log, placement, detail = simulate_tournament(tour, rng)
            champ = next(t for t, p in placement.items() if p == "1st")
            champs[champ] += 1
            print(f"## Run {i}: champion {champ}\n")
            for rec in log:
                sc = rec.score if rec.winner == rec.team_a else rec.score[::-1]
                maps = ""
                if rec.match_id in detail:
                    maps = "  [" + ", ".join(f"{m.map} {m.score[0]}-{m.score[1]}" for m in detail[rec.match_id].maps) + "]"
                status = "real" if rec.locked else "sim"
                print(f"{rec.match_id:<11} {stage_label(rec.match_id):<27} {tag(rec.team_a):>4} vs {tag(rec.team_b):<4} "
                      f"-> {rec.winner} {sc[0]}-{sc[1]} ({status}){maps}")
            print()
        if args.runs > 1:
            print("Champions across runs: " + ", ".join(f"{k} x{v}" for k, v in champs.most_common()))
    elif args.cmd == "rankings":
        print(power_rankings_md(model))
    elif args.cmd == "team":
        name = ds.find_team(args.name).name
        full = team_profiles_md(ds, model, PlayerModel(ds))
        section = full.split(f"## {name} (")[1].split("\n## ")[0]
        print(f"## {name} (" + section)
    elif args.cmd == "tournament":
        tour = Tournament(ds, model)
        print(tournament_md(ds, model, tour, tour.simulate(args.sims), PlayerModel(ds)))
    elif args.cmd == "matrix":
        teams = [r.team.name for r in model.power_rankings()]
        for x, y in combinations(teams, 2):
            p = predict_series(model, x, y, args.bo, home_region=args.home)
            s, _ = p.predicted_score()
            fs = s if p.favourite == x else s[::-1]
            print(f"{x:>18} vs {y:<18} -> {p.favourite} {fs[0]}-{fs[1]} ({pct(p.p_favourite)})")
    elif args.cmd == "sync-vlr":
        from .vlr import sync
        team = ds.find_team(args.team).name if args.team else None
        sync(team, args.timespan, args.keep_impact, args.overwrite_verified,
             data_dir=Path(args.data) if args.data else ROOT / "data")
        print("Now run: python -m vct_predictor report")
    elif args.cmd == "report":
        for p in write_reports(ds, model, Path(args.out), args.sims):
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
