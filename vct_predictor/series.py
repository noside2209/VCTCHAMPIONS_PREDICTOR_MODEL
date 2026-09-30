"""Full series prediction: veto -> per-map probabilities & scorelines -> series score."""
from __future__ import annotations

from dataclasses import dataclass, field

from .ratings import RatingModel
from .scores import expected_rounds, most_likely_score, overtime_prob
from .veto import VetoStep, run_veto


@dataclass
class MapPrediction:
    order: int
    map: str
    picked_by: str
    p_a: float
    likely_score: tuple[int, int]  # most likely exact score (overall)
    likely_score_prob: float
    fav_score: tuple[int, int]  # most likely score given the map favourite wins
    expected_rounds: tuple[float, float]
    p_overtime: float


@dataclass
class SeriesPrediction:
    team_a: str
    team_b: str
    bo: int
    pool: list[str]
    home_region: str | None
    veto: list[VetoStep]
    maps: list[MapPrediction]
    score_dist: dict[tuple[int, int], float]
    p_a: float
    all_map_probs: dict[str, float]
    h2h_adj: float
    h2h_records: list[str] = field(default_factory=list)

    @property
    def favourite(self) -> str:
        return self.team_a if self.p_a >= 0.5 else self.team_b

    @property
    def p_favourite(self) -> float:
        return max(self.p_a, 1.0 - self.p_a)

    def predicted_score(self) -> tuple[tuple[int, int], float]:
        """Most likely exact series score among outcomes where the favourite wins."""
        fav_a = self.p_a >= 0.5
        cands = [(k, v) for k, v in self.score_dist.items() if (k[0] > k[1]) == fav_a]
        return max(cands, key=lambda kv: kv[1])

    def most_likely_overall(self) -> tuple[tuple[int, int], float]:
        return max(self.score_dist.items(), key=lambda kv: kv[1])


def series_score_distribution(map_probs: list[float], bo: int) -> dict[tuple[int, int], float]:
    """Exact distribution of final series scores given per-map win probabilities (in play order)."""
    need = bo // 2 + 1
    dist: dict[tuple[int, int], float] = {}
    states = {(0, 0): 1.0}
    for pa in map_probs:
        nxt: dict[tuple[int, int], float] = {}
        for (a, b), p in states.items():
            for (na, nb), q in (((a + 1, b), pa), ((a, b + 1), 1 - pa)):
                if na == need or nb == need:
                    dist[(na, nb)] = dist.get((na, nb), 0.0) + p * q
                else:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + p * q
        states = nxt
    return dist


def predict_series(
    model: RatingModel,
    team_a: str,
    team_b: str,
    bo: int = 3,
    pool: list[str] | None = None,
    home_region: str | None = None,
    use_h2h: bool = True,
) -> SeriesPrediction:
    pool = list(pool or model.pool)
    probs = {m: model.map_win_prob(team_a, team_b, m, home_region, use_h2h) for m in pool}
    steps, played = run_veto(team_a, team_b, pool, bo, lambda m: probs[m])
    maps: list[MapPrediction] = []
    for i, (m, by) in enumerate(played, 1):
        p = probs[m]
        (ls, lp) = most_likely_score(p)
        fav, _ = most_likely_score(p, winner_is_a=p >= 0.5)
        maps.append(MapPrediction(i, m, by, p, ls, lp, fav, expected_rounds(p), overtime_prob(p)))
    dist = series_score_distribution([mp.p_a for mp in maps], bo)
    p_a = sum(v for (a, b), v in dist.items() if a > b)
    return SeriesPrediction(
        team_a, team_b, bo, pool, home_region, steps, maps, dist, p_a, probs,
        model.h2h_adjustment(team_a, team_b) if use_h2h else 0.0,
        model.h2h_records(team_a, team_b),
    )
