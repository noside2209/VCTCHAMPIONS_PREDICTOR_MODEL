"""Champions 2026 bracket: GSL groups -> 8-team double-elimination playoffs.

The same bracket runner is used for:
  * the "chalk" forecast (favourite always wins with its most likely score), and
  * Monte Carlo simulation (every series sampled from its score distribution).
Completed results from data/champions_results.json are always locked in.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from .data import Dataset
from .ratings import RatingModel
from .series import SeriesPrediction, predict_series

PLACEMENTS = ["1st", "2nd", "3rd", "4th", "5-6th", "7-8th", "9-12th", "13-16th"]

STAGE_LABELS = {
    "opener1": "Opening match", "opener2": "Opening match", "winners": "Winners' match",
    "elim": "Elimination match", "decider": "Decider match",
    "UBQF": "Upper quarterfinal", "UBSF": "Upper semifinal", "UBF": "Upper final",
    "LBR1": "Lower round 1", "LBR2": "Lower round 2", "LBR3": "Lower round 3",
    "LBF": "Lower final", "GF": "Grand final",
}


def stage_label(match_id: str) -> str:
    if "-" in match_id:
        g, kind = match_id.split("-", 1)
        return f"Group {g} {STAGE_LABELS[kind]}"
    base = match_id.split("_")[0]
    return STAGE_LABELS.get(base) or STAGE_LABELS[base.rstrip("1234")]


@dataclass
class MatchRecord:
    match_id: str
    team_a: str
    team_b: str
    bo: int
    winner: str
    loser: str
    score: tuple[int, int]  # from team_a's perspective
    locked: bool


Decider = Callable[[str, str, str, int], tuple[str, str, tuple[int, int]]]


class Tournament:
    def __init__(self, ds: Dataset, model: RatingModel):
        self.ds = ds
        self.model = model
        self.host = ds.event.get("host_region")
        self.locked: dict[str, dict] = {r["match"]: r for r in ds.results.get("series", [])}
        self._cache: dict[tuple[str, str, int], SeriesPrediction] = {}

    # ---------------------------------------------------------------- series
    def prediction(self, a: str, b: str, bo: int) -> SeriesPrediction:
        key = (a, b, bo)
        if key not in self._cache:
            self._cache[key] = predict_series(self.model, a, b, bo, home_region=self.host)
        return self._cache[key]

    def _locked_outcome(self, match_id: str, a: str, b: str):
        r = self.locked.get(match_id)
        if not r:
            return None
        if {r["team_a"], r["team_b"]} != {a, b}:
            raise ValueError(
                f"Result for {match_id} lists {r['team_a']} vs {r['team_b']} but the bracket has {a} vs {b}")
        sa, sb = r["score"] if r["team_a"] == a else tuple(reversed(r["score"]))
        return (a, b, (sa, sb)) if sa > sb else (b, a, (sa, sb))

    # ---------------------------------------------------------------- runner
    def run(self, decide: Decider) -> tuple[list[MatchRecord], dict[str, str]]:
        log: list[MatchRecord] = []
        placement: dict[str, str] = {}

        def play(mid: str, a: str, b: str, bo: int = 3) -> tuple[str, str]:
            locked = self._locked_outcome(mid, a, b)
            if locked:
                w, l, sc = locked
            else:
                w, l, sc = decide(mid, a, b, bo)
            log.append(MatchRecord(mid, a, b, bo, w, l, sc, bool(locked)))
            return w, l

        seeds: dict[str, str] = {}
        for g, openers in self.ds.event["group_openers"].items():
            w1, l1 = play(f"{g}-opener1", *openers[0])
            w2, l2 = play(f"{g}-opener2", *openers[1])
            ww, wl = play(f"{g}-winners", w1, w2)
            ew, el = play(f"{g}-elim", l1, l2)
            dw, dl = play(f"{g}-decider", wl, ew)
            seeds[f"{g}1"], seeds[f"{g}2"] = ww, dw
            placement[dl] = "9-12th"
            placement[el] = "13-16th"

        qf = self.ds.event["playoff_upper_quarterfinals"]
        qw, ql = [], []
        for i, (s1, s2) in enumerate(qf, 1):
            w, l = play(f"UBQF{i}", seeds[s1], seeds[s2])
            qw.append(w)
            ql.append(l)
        sf1w, sf1l = play("UBSF1", qw[0], qw[1])
        sf2w, sf2l = play("UBSF2", qw[2], qw[3])
        l11w, l11l = play("LBR1_1", ql[0], ql[1])
        l12w, l12l = play("LBR1_2", ql[2], ql[3])
        placement[l11l] = placement[l12l] = "7-8th"
        ubfw, ubfl = play("UBF", sf1w, sf2w)
        # Cross-over so upper-semi losers don't immediately rematch.
        l21w, l21l = play("LBR2_1", sf2l, l11w)
        l22w, l22l = play("LBR2_2", sf1l, l12w)
        placement[l21l] = placement[l22l] = "5-6th"
        l3w, l3l = play("LBR3", l21w, l22w)
        placement[l3l] = "4th"
        lfw, lfl = play("LBF", ubfl, l3w, 5)
        placement[lfl] = "3rd"
        gfw, gfl = play("GF", ubfw, lfw, 5)
        placement[gfl] = "2nd"
        placement[gfw] = "1st"
        return log, placement

    # ---------------------------------------------------------------- modes
    def chalk(self) -> tuple[list[MatchRecord], dict[str, str]]:
        def decide(mid, a, b, bo):
            pred = self.prediction(a, b, bo)
            score, _ = pred.predicted_score()
            return (a, b, score) if score[0] > score[1] else (b, a, score)
        return self.run(decide)

    def simulate(self, n: int = 20000, seed: int = 2026) -> dict:
        rng = random.Random(seed)
        place_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        pairing_counts: dict[str, dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
        dist_cache: dict[tuple[str, str, int], tuple[list, list]] = {}

        def decide(mid, a, b, bo):
            key = (a, b, bo)
            if key not in dist_cache:
                d = self.prediction(a, b, bo).score_dist
                dist_cache[key] = (list(d.keys()), list(d.values()))
            scores, weights = dist_cache[key]
            sc = rng.choices(scores, weights)[0]
            return (a, b, sc) if sc[0] > sc[1] else (b, a, sc)

        for _ in range(n):
            log, placement = self.run(decide)
            for team, pl in placement.items():
                place_counts[team][pl] += 1
            for rec in log:
                if not rec.locked:
                    pairing_counts[rec.match_id][(rec.team_a, rec.team_b)] += 1

        teams = list(self.ds.teams)
        out = {}
        for t in teams:
            c = place_counts[t]
            idx = {p: i for i, p in enumerate(PLACEMENTS)}
            cum = lambda k: sum(v for p, v in c.items() if idx[p] <= k) / n  # noqa: E731
            out[t] = {
                "placements": {p: c.get(p, 0) / n for p in PLACEMENTS},
                "win": cum(0), "final": cum(1), "top3": cum(2), "top4": cum(3),
                "playoffs": cum(5),
            }
        pairings = {mid: sorted(((k, v / n) for k, v in d.items()), key=lambda kv: -kv[1])
                    for mid, d in pairing_counts.items()}
        return {"teams": out, "pairings": pairings, "sims": n}
