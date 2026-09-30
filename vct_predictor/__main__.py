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
"""
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

from .data import load_dataset
from .players import PlayerModel
from .ratings import RatingModel
from .report import (format_series, pct, power_rankings_md, team_profiles_md, tournament_md,
                     write_reports)
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
