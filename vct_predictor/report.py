"""Text / markdown / CSV / JSON rendering."""
from __future__ import annotations

import csv
import json
from itertools import combinations, permutations
from pathlib import Path

from .data import Dataset
from .ratings import RatingModel
from .series import SeriesPrediction, predict_series
from .tournament import PLACEMENTS, Tournament, stage_label


def pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def score_str(s: tuple[int, int]) -> str:
    return f"{s[0]}-{s[1]}"


def format_series(pred: SeriesPrediction, model: RatingModel, title: str | None = None, show_pool: bool = True) -> str:
    a, b = pred.team_a, pred.team_b
    ta, tb = model.ratings.get(a), model.ratings.get(b)
    tag = lambda n: model.ratings[n].team.tag if n in model.ratings else n  # noqa: E731
    lines = []
    lines.append(f"### {title or f'{a} vs {b}'} (Bo{pred.bo})")
    venue = f"home region: {pred.home_region}" if pred.home_region else "neutral venue"
    lines.append("")
    (ps, pp) = pred.predicted_score()
    fav = pred.favourite
    fav_score = ps if fav == a else (ps[1], ps[0])
    lines.append(f"**Prediction: {fav} wins {score_str(fav_score)}** "
                 f"(series win {pct(pred.p_favourite)}; this exact score {pct(pp)}) — {venue}")
    lines.append("")
    if ta and tb:
        lines.append(f"- Ratings: {a} {ta.elo:.0f} vs {b} {tb.elo:.0f}"
                     f" | H2H adjustment: {round(pred.h2h_adj) or 0:+d} Elo for {a}")
    dist = sorted(pred.score_dist.items(), key=lambda kv: (-(kv[0][0] - kv[0][1])))
    lines.append("- Series score odds: " + ", ".join(
        f"{tag(a)} {score_str(k)} {tag(b)}: {pct(v)}" for k, v in dist))
    if pred.h2h_records:
        lines.append("- Head-to-head on record: " + "; ".join(pred.h2h_records))
    lines.append("")
    lines.append("**Map veto** (" + a + " starts): " + " → ".join(
        f"{tag(s.team)} {s.action} {s.map}" if s.action != "decider" else f"{s.map} decider"
        for s in pred.veto))
    lines.append("")
    lines.append(f"| # | Map | Picked by | {tag(a)} win | Predicted score | Most likely exact | OT chance |")
    lines.append("|---|-----|-----------|------|-----------------|-------------------|-----------|")
    for mp in pred.maps:
        fav_m = a if mp.p_a >= 0.5 else b
        fs = mp.fav_score if fav_m == a else (mp.fav_score[1], mp.fav_score[0])
        lines.append(f"| {mp.order} | {mp.map} | {tag(mp.picked_by) if mp.picked_by != 'decider' else 'decider'} "
                     f"| {pct(mp.p_a)} | {tag(fav_m)} {score_str(fs)} "
                     f"| {tag(a)} {score_str(mp.likely_score)} {tag(b)} ({pct(mp.likely_score_prob)}) "
                     f"| {pct(mp.p_overtime)} |")
    if show_pool:
        lines.append("")
        lines.append("Map-by-map win chance for " + a + ": " + ", ".join(
            f"{m} {pct(p)}" for m, p in sorted(pred.all_map_probs.items(), key=lambda kv: -kv[1])))
    lines.append("")
    return "\n".join(lines)


def power_rankings_md(model: RatingModel) -> str:
    lines = ["| Rank | Team | Region | Seed | Rating | Prior | Roster | Form | Best maps | Worst map |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(model.power_rankings(), 1):
        maps = model.best_maps(r.team.name)
        best = ", ".join(m for m, _ in maps[:2])
        worst = maps[-1][0]
        lines.append(f"| {i} | {r.team.name} | {r.team.region} | {r.team.seed} | {r.elo:.0f} | {r.prior:.0f} "
                     f"| {r.roster_adj:+.0f} | {r.form:+.0f} | {best} | {worst} |")
    return "\n".join(lines)


def team_profiles_md(ds: Dataset, model: RatingModel) -> str:
    out = ["# Team profiles", "",
           "Player impact values are analyst estimates on a VLR-rating-like scale (1.00 = average VCT starter). "
           "Map ratings are Elo offsets (positive = comfort map).", ""]
    pool = model.pool
    for r in model.power_rankings():
        t = r.team
        out.append(f"## {t.name} ({t.tag}) — {t.seed}")
        out.append("")
        out.append(f"Rating **{r.elo:.0f}** (prior {r.prior:.0f}, roster {r.roster_adj:+.1f}, form {r.form:+.1f})")
        out.append("")
        out.append("Roster: " + ", ".join(f"{p.name} ({p.impact:.2f})" for p in t.players)
                   + (f" — bench: {', '.join(t.bench)}" if t.bench else ""))
        out.append("")
        if t.resume_2026:
            out.append("2026 résumé: " + "; ".join(t.resume_2026))
            out.append("")
        out.append("| " + " | ".join(pool) + " |")
        out.append("|" + "---|" * len(pool))
        out.append("| " + " | ".join(f"{r.map_ratings[m]:+.0f}" for m in pool) + " |")
        out.append("")
        if r.series_log:
            out.append("Logged series: " + "; ".join(r.series_log))
            out.append("")
    return "\n".join(out)


def tournament_md(ds: Dataset, model: RatingModel, tour: Tournament, sims: dict) -> str:
    ev = ds.event
    log, placement = tour.chalk()
    out = [f"# {ev['name']} forecast — {ev['location']}", "",
           f"Data as of **{ev['data_as_of']}**. Map pool: {', '.join(ev['map_pool'])}. "
           f"Monte Carlo: {sims['sims']:,} simulations. Home region bonus for {ev['host_region']} teams: "
           f"+{ev['home_advantage_elo']} Elo.", ""]
    out += ["## Title odds", "",
            "| Team | Playoffs | Top 4 | Final | Champion | Most likely finish |",
            "|---|---|---|---|---|---|"]
    for t, v in sorted(sims["teams"].items(), key=lambda kv: (-kv[1]["win"], -kv[1]["playoffs"])):
        likely = max(v["placements"].items(), key=lambda kv: kv[1])
        out.append(f"| {t} | {pct(v['playoffs'])} | {pct(v['top4'])} | {pct(v['final'])} | **{pct(v['win'])}** "
                   f"| {likely[0]} ({pct(likely[1])}) |")
    out += ["", "## Power rankings", "", power_rankings_md(model), ""]

    out += ["## Predicted bracket (most likely path)", "",
            "Every match below uses the favourite with its single most likely exact score. "
            "Locked results are real.", ""]
    out += ["| Match | Stage | Team A | Team B | Result | Status |", "|---|---|---|---|---|---|"]
    for rec in log:
        out.append(f"| {rec.match_id} | {stage_label(rec.match_id)} | {rec.team_a} | {rec.team_b} "
                   f"| **{rec.winner}** {score_str(rec.score if rec.winner == rec.team_a else rec.score[::-1])} "
                   f"| {'FINAL (real result)' if rec.locked else 'predicted'} |")
    out.append("")
    champion = next(t for t, p in placement.items() if p == "1st")
    out.append(f"**Predicted champion: {champion}**")
    out.append("")
    out.append("Predicted final standings: " + "; ".join(
        f"{p}: " + ", ".join(t for t, pp in placement.items() if pp == p) for p in PLACEMENTS))
    out.append("")

    out += ["## Possible opponents for upcoming matches", "",
            "Probability of each pairing across all simulations (top 4 shown per match).", ""]
    for mid, pairs in sims["pairings"].items():
        out.append(f"- **{mid}** ({stage_label(mid)}): " + "; ".join(
            f"{a} vs {b} {pct(p)}" for (a, b), p in pairs[:4]))
    out.append("")

    out += ["## Match-by-match breakdown (predicted path)", ""]
    for rec in log:
        if rec.locked:
            continue
        pred = tour.prediction(rec.team_a, rec.team_b, rec.bo)
        out.append(format_series(pred, model, f"{rec.match_id} — {stage_label(rec.match_id)}: "
                                              f"{rec.team_a} vs {rec.team_b}"))
    return "\n".join(out)


def all_matchups_md(ds: Dataset, model: RatingModel, bo: int, home_region: str | None = None) -> str:
    teams = [r.team.name for r in model.power_rankings()]
    out = [f"# Every possible matchup — Bo{bo}", "",
           f"{len(teams) * (len(teams) - 1) // 2} pairings. Venue: "
           f"{'home region ' + home_region if home_region else 'neutral (no home advantage)'}. "
           "The higher-rated team is listed first and starts the veto.", "",
           "## Quick table", "",
           "| Team A | Team B | Favourite | Win % | Predicted score | Maps (veto order) |",
           "|---|---|---|---|---|---|"]
    preds = []
    for a, b in combinations(teams, 2):
        p = predict_series(model, a, b, bo, home_region=home_region)
        preds.append(p)
        (s, _) = p.predicted_score()
        fs = s if p.favourite == a else s[::-1]
        out.append(f"| {a} | {b} | {p.favourite} | {pct(p.p_favourite)} | {score_str(fs)} "
                   f"| {', '.join(m.map for m in p.maps)} |")
    out += ["", "## Full breakdowns", ""]
    for p in preds:
        out.append(format_series(p, model))
    return "\n".join(out)


def write_matrix_csv(model: RatingModel, bo: int, path: Path, home_region: str | None = None) -> None:
    teams = [r.team.name for r in model.power_rankings()]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([f"Row team win % vs column (Bo{bo})"] + teams)
        for a in teams:
            row = [a]
            for b in teams:
                row.append("" if a == b else f"{100 * predict_series(model, a, b, bo, home_region=home_region).p_a:.1f}")
            w.writerow(row)


def write_map_csv(model: RatingModel, path: Path, home_region: str | None = None) -> None:
    teams = [r.team.name for r in model.power_rankings()]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["team_a", "team_b"] + model.pool)
        for a, b in permutations(teams, 2):
            w.writerow([a, b] + [f"{100 * model.map_win_prob(a, b, m, home_region):.1f}" for m in model.pool])


def series_json(pred: SeriesPrediction) -> dict:
    ps, pp = pred.predicted_score()
    return {
        "a": pred.team_a, "b": pred.team_b, "bo": pred.bo, "pA": round(pred.p_a, 4),
        "pred": list(ps), "predP": round(pp, 4), "h2h": round(pred.h2h_adj, 1), "records": pred.h2h_records,
        "dist": [[k[0], k[1], round(v, 4)] for k, v in sorted(pred.score_dist.items(), key=lambda kv: kv[0][1] - kv[0][0])],
        "veto": [[s.team, s.action, s.map] for s in pred.veto],
        "maps": [{"map": m.map, "by": m.picked_by, "pA": round(m.p_a, 4), "likely": list(m.likely_score),
                  "likelyP": round(m.likely_score_prob, 4), "fav": list(m.fav_score), "ot": round(m.p_overtime, 4),
                  "exp": [round(m.expected_rounds[0], 1), round(m.expected_rounds[1], 1)]} for m in pred.maps],
        "pool": {k: round(v, 4) for k, v in pred.all_map_probs.items()},
    }


def dashboard_data(ds: Dataset, model: RatingModel, tour: Tournament, sims: dict) -> dict:
    """Everything the HTML dashboard needs: team profiles, all matchups (neutral), tournament forecast."""
    teams = [r.team.name for r in model.power_rankings()]
    matchups = {}
    for bo in (1, 3, 5):
        for a, b in permutations(teams, 2):
            matchups[f"{a}|{b}|{bo}"] = series_json(predict_series(model, a, b, bo))
    log, placement = tour.chalk()
    bracket = []
    for rec in log:
        entry = {"id": rec.match_id, "stage": stage_label(rec.match_id), "a": rec.team_a, "b": rec.team_b,
                 "bo": rec.bo, "winner": rec.winner, "score": list(rec.score), "locked": rec.locked}
        if not rec.locked:
            entry["detail"] = series_json(tour.prediction(rec.team_a, rec.team_b, rec.bo))
        bracket.append(entry)
    return {
        "event": {k: ds.event[k] for k in ("name", "location", "data_as_of", "map_pool", "host_region", "dates")},
        "teams": [{
            "name": r.team.name, "tag": r.team.tag, "region": r.team.region, "seed": r.team.seed,
            "elo": round(r.elo, 1), "prior": r.prior, "roster": round(r.roster_adj, 1), "form": round(r.form, 1),
            "players": [[p.name, p.impact] for p in r.team.players], "resume": r.team.resume_2026,
            "maps": {m: round(v, 1) for m, v in r.map_ratings.items()},
        } for r in model.power_rankings()],
        "matchups": matchups,
        "bracket": bracket,
        "placement": placement,
        "odds": sims["teams"],
        "sims": sims["sims"],
    }


def write_reports(ds: Dataset, model: RatingModel, out_dir: Path, sims_n: int = 20000) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tour = Tournament(ds, model)
    sims = tour.simulate(sims_n)
    written = []

    def w(name: str, text: str) -> None:
        p = out_dir / name
        p.write_text(text, encoding="utf-8")
        written.append(p)

    w("tournament_forecast.md", tournament_md(ds, model, tour, sims))
    w("team_profiles.md", team_profiles_md(ds, model))
    for bo in (1, 3, 5):
        w(f"all_matchups_bo{bo}.md", all_matchups_md(ds, model, bo))
        write_matrix_csv(model, bo, out_dir / f"matchup_matrix_bo{bo}.csv")
        written.append(out_dir / f"matchup_matrix_bo{bo}.csv")
    write_map_csv(model, out_dir / "map_win_probabilities.csv")
    written.append(out_dir / "map_win_probabilities.csv")
    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps(dashboard_data(ds, model, tour, sims), separators=(",", ":")).replace("</", "<\\/")
    w("dashboard.html", template.replace("/*__DATA__*/null", payload))
    return written
