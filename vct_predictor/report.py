"""Text / markdown / CSV / JSON rendering."""
from __future__ import annotations

import csv
import json
from itertools import combinations, permutations
from pathlib import Path

from .data import Dataset
from .players import PlayerModel, StatLine
from .ratings import RatingModel
from .series import SeriesPrediction, predict_series
from .sides import PISTOL_PRIOR_WEIGHT as PISTOL_W
from .tournament import PLACEMENTS, Tournament, stage_label


def pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def score_str(s: tuple[int, int]) -> str:
    return f"{s[0]}-{s[1]}"


STAT_HEADER = "| Player | Team | Agent | K | D | A | K/D | ACS | ADR | KAST | FK | FD | Rating |"
STAT_RULE = "|---|---|---|---|---|---|---|---|---|---|---|---|---|"


def _n(v: float) -> str:
    return str(v) if isinstance(v, int) else f"{v:.1f}"


def _stat_row(ln: StatLine, tag: str, agent: str | None = None) -> str:
    return (f"| {ln.player} | {tag} | {agent if agent is not None else ln.agent} | {_n(ln.kills)} | {_n(ln.deaths)} "
            f"| {_n(ln.assists)} | {ln.kd:.2f} | {ln.acs:.0f} | {ln.adr:.0f} | {100 * ln.kast:.0f}% "
            f"| {_n(ln.fk)} | {_n(ln.fd)} | {ln.rating:.2f} |")


def format_players(pred: SeriesPrediction, players: PlayerModel, model: RatingModel, per_map: bool = True) -> str:
    tag = lambda n: model.ratings[n].team.tag if n in model.ratings else n  # noqa: E731
    proj = players.project_series(pred)
    out = []
    if per_map:
        for mp in proj["maps"]:
            out.append(f"**{mp['map']}: projected box score** (chance this map is played: {pct(mp['p_played'])})")
            out.append("")
            out += [STAT_HEADER, STAT_RULE]
            for ln in sorted(mp["a"], key=lambda x: -x.rating) + sorted(mp["b"], key=lambda x: -x.rating):
                out.append(_stat_row(ln, tag(ln.team)))
            out.append("")
    out.append(f"**Series projection** (expected totals over {proj['expected_maps']:.2f} maps, weighted by the chance each map is played)")
    out.append("")
    out += [STAT_HEADER.replace("| Agent ", "| Role "), STAT_RULE]
    for ln in sorted(proj["series"], key=lambda x: (x.team != pred.team_a, -x.rating)):
        out.append(_stat_row(ln, tag(ln.team), ln.role))
    out.append("")
    mvp, top = proj["mvp"], proj["top_fragger"]
    out.append(f"Projected MVP: **{mvp.player}** ({tag(mvp.team)}, {mvp.rating:.2f} rating). "
               f"Projected top fragger: **{top.player}** ({top.kills:.1f} kills).")
    out.append("")
    return "\n".join(out)


def format_series(pred: SeriesPrediction, model: RatingModel, title: str | None = None, show_pool: bool = True,
                  players: PlayerModel | None = None, player_detail: str = "full") -> str:
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
    lines.append(f"| # | Map | Picked by | {tag(a)} win | Predicted score | Most likely exact | Blowout | OT chance "
                 f"| {tag(a)} atk pistol | {tag(a)} def pistol |")
    lines.append("|---|-----|-----------|------|-----------------|-------------------|---------|-----------|------|------|")
    for mp in pred.maps:
        fav_m = a if mp.p_a >= 0.5 else b
        fs = mp.fav_score if fav_m == a else (mp.fav_score[1], mp.fav_score[0])
        lines.append(f"| {mp.order} | {mp.map} | {tag(mp.picked_by) if mp.picked_by != 'decider' else 'decider'} "
                     f"| {pct(mp.p_a)} | {tag(fav_m)} {score_str(fs)} "
                     f"| {tag(a)} {score_str(mp.likely_score)} {tag(b)} ({pct(mp.likely_score_prob)}) "
                     f"| {pct(mp.p_blowout)} | {pct(mp.p_overtime)} | {pct(mp.p_pistol_atk)} | {pct(mp.p_pistol_def)} |")
    if show_pool:
        lines.append("")
        lines.append("Map-by-map win chance for " + a + ": " + ", ".join(
            f"{m} {pct(p)}" for m, p in sorted(pred.all_map_probs.items(), key=lambda kv: -kv[1])))
    lines.append("")
    if players is not None and player_detail != "none":
        lines.append(format_players(pred, players, model, per_map=player_detail == "full"))
    return "\n".join(lines)


def format_sim_series(sim, model: RatingModel, title: str | None = None, box_scores: bool = True) -> str:
    tag = lambda n: model.ratings[n].team.tag if n in model.ratings else n  # noqa: E731
    a, b = sim.team_a, sim.team_b
    sw = sim.score if sim.winner == a else sim.score[::-1]
    lines = [f"### {title or f'{a} vs {b}'} (Bo{sim.bo})", "",
             f"**{sim.winner} wins {sw[0]}-{sw[1]}**", "",
             "Veto: " + " → ".join(f"{tag(s.team)} {s.action} {s.map}" if s.action != "decider"
                                   else f"{s.map} decider" for s in sim.veto), ""]
    for i, m in enumerate(sim.maps, 1):
        w = a if m.score[0] > m.score[1] else b
        flavour = " (overtime)" if max(m.score) > 13 else " (blowout)" if min(m.score) <= 5 else ""
        by = "decider" if m.picked_by == "decider" else f"{tag(m.picked_by)} pick"
        pist = ", ".join(f"R{n} {tag(a) if won else tag(b)} ({'atk' if (side == 'atk') == won else 'def'})"
                         for n, won, side in m.pistols)
        half = f", half {m.half[0]}-{m.half[1]}" if m.pistols else ""
        lines.append(f"- Map {i} {m.map} ({by}): {tag(a)} {m.score[0]}-{m.score[1]} {tag(b)} → {tag(w)}{flavour}"
                     + (f"  · pistols: {pist}{half}" if pist else ""))
    if sim.unplayed:
        lines.append(f"- Not played: {', '.join(sim.unplayed)}")
    lines.append("")
    if box_scores and sim.maps and sim.maps[0].lines_a:
        for m in sim.maps:
            lines.append(f"**{m.map} box score** ({tag(a)} {m.score[0]}-{m.score[1]} {tag(b)})")
            lines.append("")
            lines += [STAT_HEADER, STAT_RULE]
            for ln in sorted(m.lines_a, key=lambda x: -x.rating) + sorted(m.lines_b, key=lambda x: -x.rating):
                lines.append(_stat_row(ln, tag(ln.team)))
            lines.append("")
        mvp = sim.mvp()
        if mvp:
            lines.append(f"Series MVP: **{mvp.player}** ({tag(mvp.team)})")
            lines.append("")
    return "\n".join(lines)


def pistol_rows(model: RatingModel, m: str) -> list[dict]:
    """Every team's attack/defence pistol and round win rates on a map, best pistol team first."""
    rows = []
    for t in model.ratings:
        r = model.sides.rates(t, m)
        rows.append({"team": t, "tag": model.ratings[t].team.tag,
                     "atk": r.pistol["atk"], "def": r.pistol["def"],
                     "atk_raw": r.pistol_raw["atk"], "def_raw": r.pistol_raw["def"],
                     "atk_rounds": r.rounds["atk"], "def_rounds": r.rounds["def"],
                     "rounds_raw": (r.rounds_raw["atk"], r.rounds_raw["def"]),
                     "lean": model.sides.lean(t, m), "data": r.has_data})
    return sorted(rows, key=lambda x: -(x["atk"] + x["def"]))


def pistols_md(model: RatingModel, maps: list[str] | None = None) -> str:
    sm = model.sides
    out = ["# Pistol rounds by map", ""]
    if not sm.has_any_data:
        out += ["> **No real pistol data loaded yet.** Every number below is a *model estimate* from overall team",
                "> strength, so the order simply follows the power rankings. Run `python -m vct_predictor sync-vlr-sides`",
                "> on a computer that can reach vlr.gg to load each team's actual 2026 pistol and side records.", ""]
    else:
        out += [f"Source: {sm.source}. Rates are shrunk toward each team's strength-based expectation when the sample is small "
                f"(worth {PISTOL_W:.0f} pistols); raw records are shown as won/played.", ""]
    fmt = lambda rate, raw: f"{pct(rate)}" + (f" ({raw[0]}/{raw[1]})" if raw else " (est.)")  # noqa: E731
    for m in maps or model.pool:
        bias = sm.map_atk_rate(m)
        out += [f"## {m}", "", f"League attack round win rate on {m}: {pct(bias)}" + ("" if m in sm.map_atk else " (no data, assumed even)"), "",
                "| # | Team | Attack pistol | Defence pistol | Both-pistol avg | Attack rounds | Defence rounds | Side lean |",
                "|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(pistol_rows(model, m), 1):
            lean = r["lean"]
            lean_s = "even" if abs(lean) < 0.005 else (f"attack +{100 * lean:.1f}" if lean > 0 else f"defence +{-100 * lean:.1f}")
            ra, rd = r["rounds_raw"]
            out.append(f"| {i} | {r['team']} | {fmt(r['atk'], r['atk_raw'])} | {fmt(r['def'], r['def_raw'])} "
                       f"| {pct((r['atk'] + r['def']) / 2)} | {fmt(r['atk_rounds'], ra)} | {fmt(r['def_rounds'], rd)} | {lean_s} |")
        out.append("")
    return "\n".join(out)


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


def team_profiles_md(ds: Dataset, model: RatingModel, players: PlayerModel | None = None) -> str:
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
        if players is None:
            out.append("Roster: " + ", ".join(f"{p.name} ({p.impact:.2f})" for p in t.players))
        else:
            out.append("| Player | Role | IGL | Impact | Agent pool | Role info |")
            out.append("|---|---|---|---|---|---|")
            for p in players.roster(t.name):
                out.append(f"| {p.name} | {p.role} | {'yes' if p.igl else ''} | {p.impact:.2f} "
                           f"| {', '.join(p.agents)} | {p.source} |")
            out.append("")
            out.append("Projected agents by map:")
            out.append("")
            out.append("| Player | " + " | ".join(pool) + " |")
            out.append("|---|" + "---|" * len(pool))
            by_map = {m: players.assign_agents(t.name, m) for m in pool}
            for p in players.roster(t.name):
                out.append(f"| {p.name} | " + " | ".join(by_map[m][p.name] for m in pool) + " |")
        if t.bench:
            out.append("")
            out.append(f"Bench: {', '.join(t.bench)}")
        out.append("")
        if t.resume_2026:
            out.append("2026 résumé: " + "; ".join(t.resume_2026))
            out.append("")
        out.append("Map ratings (Elo offset):")
        out.append("")
        out.append("| " + " | ".join(pool) + " |")
        out.append("|" + "---|" * len(pool))
        out.append("| " + " | ".join(f"{r.map_ratings[m]:+.0f}" for m in pool) + " |")
        out.append("")
        if r.series_log:
            out.append("Logged series: " + "; ".join(r.series_log))
            out.append("")
    return "\n".join(out)


def tournament_md(ds: Dataset, model: RatingModel, tour: Tournament, sims: dict,
                  players: PlayerModel | None = None) -> str:
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
                                              f"{rec.team_a} vs {rec.team_b}", players=players))
    return "\n".join(out)


def all_matchups_md(ds: Dataset, model: RatingModel, bo: int, home_region: str | None = None,
                    players: PlayerModel | None = None) -> str:
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
        out.append(format_series(p, model, players=players, player_detail="series"))
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


def players_json(pred: SeriesPrediction, players: PlayerModel) -> dict:
    proj = players.project_series(pred)
    return {
        "maps": [{"map": m["map"], "p": round(m["p_played"], 3),
                  "a": [ln.as_list() for ln in m["a"]], "b": [ln.as_list() for ln in m["b"]]} for m in proj["maps"]],
        "series": [[ln.team == pred.team_a, ln.role] + ln.as_list() for ln in proj["series"]],
        "mvp": proj["mvp"].player, "top": proj["top_fragger"].player, "expMaps": round(proj["expected_maps"], 2),
    }


def series_json(pred: SeriesPrediction, players: PlayerModel | None = None) -> dict:
    ps, pp = pred.predicted_score()
    extra = {"players": players_json(pred, players)} if players is not None else {}
    return {**extra,
        "a": pred.team_a, "b": pred.team_b, "bo": pred.bo, "pA": round(pred.p_a, 4),
        "pred": list(ps), "predP": round(pp, 4), "h2h": round(pred.h2h_adj, 1), "records": pred.h2h_records,
        "dist": [[k[0], k[1], round(v, 4)] for k, v in sorted(pred.score_dist.items(), key=lambda kv: kv[0][1] - kv[0][0])],
        "veto": [[s.team, s.action, s.map] for s in pred.veto],
        "maps": [{"map": m.map, "by": m.picked_by, "pA": round(m.p_a, 4), "likely": list(m.likely_score),
                  "likelyP": round(m.likely_score_prob, 4), "fav": list(m.fav_score), "ot": round(m.p_overtime, 4),
                  "blow": round(m.p_blowout, 4), "pistA": round(m.p_pistol_atk, 4), "pistD": round(m.p_pistol_def, 4),
                  "exp": [round(m.expected_rounds[0], 1), round(m.expected_rounds[1], 1)]} for m in pred.maps],
        "pool": {k: round(v, 4) for k, v in pred.all_map_probs.items()},
        "ctx": ctx_json(pred.team_a, pred.team_b, pred.pool, _CTX_MODEL[0], pred.home_region) if _CTX_MODEL else {},
    }


_CTX_MODEL: list = []


def ctx_json(a: str, b: str, pool: list[str], model: RatingModel, home: str | None) -> dict:
    """Per-map [ratings-only map chance, A atk pistol, A def pistol, atk shift, def shift] for the browser simulator.

    Pistol entries are null when neutral (the simulator then derives them from strength)."""
    out = {}
    for m in pool:
        c = model.map_context(a, b, m)
        out[m] = [round(model.elo_map_prob(a, b, m, home), 4),
                  None if c.pistol_atk is None else round(c.pistol_atk, 4),
                  None if c.pistol_def is None else round(c.pistol_def, 4),
                  round(c.atk_shift, 4), round(c.def_shift, 4)]
    return out


def dashboard_data(ds: Dataset, model: RatingModel, tour: Tournament, sims: dict, players: PlayerModel) -> dict:
    """Everything the HTML dashboard needs: team profiles, all matchups (neutral), tournament forecast."""
    _CTX_MODEL[:] = [model]
    teams = [r.team.name for r in model.power_rankings()]
    matchups = {}
    for bo in (1, 3, 5):
        for a, b in permutations(teams, 2):
            matchups[f"{a}|{b}|{bo}"] = series_json(predict_series(model, a, b, bo), players)
    log, placement = tour.chalk()
    bracket = []
    for rec in log:
        entry = {"id": rec.match_id, "stage": stage_label(rec.match_id), "a": rec.team_a, "b": rec.team_b,
                 "bo": rec.bo, "winner": rec.winner, "score": list(rec.score), "locked": rec.locked}
        if not rec.locked:
            entry["detail"] = series_json(tour.prediction(rec.team_a, rec.team_b, rec.bo), players)
        bracket.append(entry)
    from .scores import CONV_ANTI_ECO, CONV_BONUS, PISTOL_SKILL, ROUND_SD, Z_CAP, r0_for_p
    from .veto import VETO_TEMPERATURE, veto_sequence
    pool = model.pool
    sim = {
        "sd": ROUND_SD, "zcap": Z_CAP, "temp": VETO_TEMPERATURE,
        "r0": [round(r0_for_p(i / 100), 5) for i in range(0, 101)],
        "veto": {bo: veto_sequence(bo, len(pool)) for bo in (1, 3, 5)},
        "weights": {t: {m: [[w["name"], w["agent"], w["role"]] + [round(w[k], 4) for k in ("kpr", "dpr", "apr", "fkpr", "fdpr")]
                            for w in players.weights(t, m)] for m in pool} for t in teams},
        "tourP": {f"{a}|{b}": [round(model.map_win_prob(a, b, m, tour.host), 4) for m in pool]
                  for a, b in permutations(teams, 2)},
        "tourCtx": {f"{a}|{b}": ctx_json(a, b, pool, model, tour.host) for a, b in permutations(teams, 2)},
        "pistolSkill": PISTOL_SKILL, "conv": [CONV_ANTI_ECO, CONV_BONUS],
        "pistols": {m: [[r["team"], round(r["atk"], 4), round(r["def"], 4), r["atk_raw"], r["def_raw"],
                         round(r["lean"], 4), r["data"]] for r in pistol_rows(model, m)] for m in pool},
        "pistolData": model.sides.has_any_data,
        "mapAtk": {m: round(model.sides.map_atk_rate(m), 4) for m in pool},
        "openers": ds.event["group_openers"],
        "qf": ds.event["playoff_upper_quarterfinals"],
        "locked": {r["match"]: [r["team_a"], r["team_b"], r["score"][0], r["score"][1]]
                   for r in ds.results.get("series", [])},
    }
    return {
        "sim": sim,
        "event": {k: ds.event[k] for k in ("name", "location", "data_as_of", "map_pool", "host_region", "dates")},
        "teams": [{
            "name": r.team.name, "tag": r.team.tag, "region": r.team.region, "seed": r.team.seed,
            "elo": round(r.elo, 1), "prior": r.prior, "roster": round(r.roster_adj, 1), "form": round(r.form, 1),
            "players": [[p.name, p.impact, p.role, p.igl, p.agents, p.source,
                         [players.assign_agents(r.team.name, m)[p.name] for m in model.pool]]
                        for p in players.roster(r.team.name)],
            "resume": r.team.resume_2026,
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
    players = PlayerModel(ds)
    written = []

    def w(name: str, text: str) -> None:
        p = out_dir / name
        p.write_text(text, encoding="utf-8")
        written.append(p)

    w("tournament_forecast.md", tournament_md(ds, model, tour, sims, players))
    w("team_profiles.md", team_profiles_md(ds, model, players))
    w("pistols.md", pistols_md(model))
    for bo in (1, 3, 5):
        w(f"all_matchups_bo{bo}.md", all_matchups_md(ds, model, bo, players=players))
        write_matrix_csv(model, bo, out_dir / f"matchup_matrix_bo{bo}.csv")
        written.append(out_dir / f"matchup_matrix_bo{bo}.csv")
    write_map_csv(model, out_dir / "map_win_probabilities.csv")
    written.append(out_dir / "map_win_probabilities.csv")
    template = (Path(__file__).parent / "dashboard_template.html").read_text(encoding="utf-8")
    payload = json.dumps(dashboard_data(ds, model, tour, sims, players), separators=(",", ":")).replace("</", "<\\/")
    w("dashboard.html", template.replace("/*__DATA__*/null", payload))
    return written
