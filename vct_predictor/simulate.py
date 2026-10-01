"""Random play-outs: one possible version of a match or of the whole event.

Unlike the prediction (the single most likely outcome), each run samples the veto,
every map round by round, and every player's box score, so upsets, blowouts and
overtime maps show up as often as the model thinks they should.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .players import PlayerModel, StatLine
from .ratings import RatingModel
from .scores import play_map
from .tournament import MatchRecord, Tournament
from .veto import VetoStep, run_veto


@dataclass
class SimMap:
    map: str
    picked_by: str
    p_a: float
    score: tuple[int, int]
    half: tuple[int, int] = (0, 0)
    a_started_atk: bool = True
    pistols: list[tuple[int, bool, str]] = field(default_factory=list)  # (round, A won?, A's side)
    lines_a: list[StatLine] = field(default_factory=list)
    lines_b: list[StatLine] = field(default_factory=list)


@dataclass
class SimSeries:
    team_a: str
    team_b: str
    bo: int
    veto: list[VetoStep]
    maps: list[SimMap]
    score: tuple[int, int]
    unplayed: list[str]

    @property
    def winner(self) -> str:
        return self.team_a if self.score[0] > self.score[1] else self.team_b

    def mvp(self) -> StatLine | None:
        lines = [ln for m in self.maps for ln in (m.lines_a if self.winner == self.team_a else m.lines_b)]
        if not lines:
            return None
        by: dict[str, list[StatLine]] = {}
        for ln in lines:
            by.setdefault(ln.player, []).append(ln)
        # Round-weighted rating across the maps played.
        best = max(by.values(), key=lambda ls: sum(x.rating * x.rounds for x in ls) / sum(x.rounds for x in ls))
        return best[0]


def simulate_series(model: RatingModel, a: str, b: str, bo: int = 3, pool: list[str] | None = None,
                    home_region: str | None = None, rng: random.Random | None = None,
                    players: PlayerModel | None = None) -> SimSeries:
    rng = rng or random.Random()
    pool = list(pool or model.pool)
    probs = {m: model.map_win_prob(a, b, m, home_region) for m in pool}
    elo = {m: model.elo_map_prob(a, b, m, home_region) for m in pool}
    steps, played = run_veto(a, b, pool, bo, lambda m: probs[m], rng=rng)
    need = bo // 2 + 1
    wa = wb = 0
    maps: list[SimMap] = []
    for m, by in played:
        if wa == need or wb == need:
            break
        play = play_map(elo[m], rng, model.map_context(a, b, m))
        sa, sb = play.score
        sm = SimMap(m, by, probs[m], (sa, sb), play.half, play.a_started_atk, play.pistols)
        if players is not None:
            sm.lines_a, sm.lines_b = players.sample_map(a, b, m, sa, sb, rng)
        maps.append(sm)
        wa += sa > sb
        wb += sb > sa
    unplayed = [m for m, _ in played[len(maps):]]
    return SimSeries(a, b, bo, steps, maps, (wa, wb), unplayed)


def simulate_tournament(tour: Tournament, rng: random.Random | None = None) -> tuple[list[MatchRecord], dict[str, str], dict[str, SimSeries]]:
    rng = rng or random.Random()
    detail: dict[str, SimSeries] = {}

    def decide(mid, a, b, bo):
        s = simulate_series(tour.model, a, b, bo, home_region=tour.host, rng=rng)
        detail[mid] = s
        return (a, b, s.score) if s.score[0] > s.score[1] else (b, a, s.score)

    log, placement = tour.run(decide)
    return log, placement, detail
