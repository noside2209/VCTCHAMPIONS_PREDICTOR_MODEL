"""Team strength model.

Strength of team T on map M against opponent O:

    rating(T) = prior_elo + roster_adj + form (Elo updates from logged series)
    map_rating(T, M) = map_delta(T, M) * familiarity(M) + map-specific Elo updates
    edge = rating(T) + map_rating(T, M) + home(T) + h2h(T, O)  -  (same for O)
    P(T wins map M) = 1 / (1 + 10 ** (-edge / 400))
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from .data import Dataset, Team

ELO_SCALE = 300.0  # tuned so e.g. an Americas #1 beats a China #4 ~85% of the time in a Bo3

# Roster adjustment: how much the (estimated) player impact shifts team Elo.
ROSTER_MEAN_BASE = 1.05
ROSTER_MEAN_COEF = 300.0
ROSTER_STAR_BASE = 1.10
ROSTER_STAR_COEF = 100.0

# Form updates.
K_MAP_HISTORY = 10.0
K_MAP_CHAMPIONS = 14.0
K_MAP_SPECIFIC = 8.0
RECENCY_HALF_LIFE_DAYS = 120.0

# Head-to-head: Elo per net (recency-weighted) series win, capped.
H2H_PER_SERIES = 10.0
H2H_CAP = 25.0
H2H_AGGREGATE_WEIGHT = 0.5

# Maps with little recent pro play: team-specific map deltas are shrunk.
MAP_FAMILIARITY = {"Abyss": 0.6, "Summit": 0.85}


def expected(edge: float) -> float:
    return 1.0 / (1.0 + 10 ** (-edge / ELO_SCALE))


@dataclass
class TeamRating:
    team: Team
    prior: float
    roster_adj: float
    form: float = 0.0
    map_ratings: dict[str, float] = field(default_factory=dict)
    series_log: list[str] = field(default_factory=list)

    @property
    def elo(self) -> float:
        return self.prior + self.roster_adj + self.form


def roster_adjustment(team: Team) -> float:
    impacts = [p.impact for p in team.players]
    if not impacts:
        return 0.0
    mean = sum(impacts) / len(impacts)
    star = max(impacts)
    return (mean - ROSTER_MEAN_BASE) * ROSTER_MEAN_COEF + max(0.0, star - ROSTER_STAR_BASE) * ROSTER_STAR_COEF


class RatingModel:
    def __init__(self, ds: Dataset, as_of: str | None = None):
        self.ds = ds
        self.as_of = date.fromisoformat(as_of or ds.event.get("data_as_of", date.today().isoformat()))
        self.pool = ds.map_pool
        self.ratings: dict[str, TeamRating] = {}
        for t in ds.teams.values():
            maps = {m: t.map_delta.get(m, 0.0) * MAP_FAMILIARITY.get(m, 1.0) for m in self.pool}
            self.ratings[t.name] = TeamRating(t, t.prior_elo, roster_adjustment(t), 0.0, maps)
        self._h2h: dict[tuple[str, str], float] = {}
        self._h2h_records: dict[tuple[str, str], list[str]] = {}
        self._apply_series(ds.history.get("series", []), K_MAP_HISTORY, "history")
        self._apply_series(ds.results.get("series", []), K_MAP_CHAMPIONS, "champions")
        for agg in ds.history.get("aggregate_h2h", []):
            net = (agg["a_wins"] - agg["b_wins"]) * H2H_AGGREGATE_WEIGHT
            self._add_h2h(agg["team_a"], agg["team_b"], net,
                          f"{agg['team_a']} {agg['a_wins']}-{agg['b_wins']} {agg['team_b']} ({agg['period']})")

    # ------------------------------------------------------------------ helpers
    def _weight(self, when: str | None) -> float:
        if not when:
            return 1.0
        days = max(0, (self.as_of - date.fromisoformat(when)).days)
        return 0.5 ** (days / RECENCY_HALF_LIFE_DAYS)

    def _base(self, name: str) -> float:
        if name in self.ratings:
            return self.ratings[name].elo
        return self.ds.external.get(name, 1750.0)

    def _map_r(self, name: str, m: str | None) -> float:
        if name in self.ratings and m in self.ratings[name].map_ratings:
            return self.ratings[name].map_ratings[m]
        return 0.0

    def _add_h2h(self, a: str, b: str, net_for_a: float, record: str) -> None:
        key = tuple(sorted((a, b)))
        sign = 1.0 if key[0] == a else -1.0
        self._h2h[key] = self._h2h.get(key, 0.0) + sign * net_for_a
        self._h2h_records.setdefault(key, []).append(record)

    def _apply_series(self, series: list[dict], k_map: float, source: str) -> None:
        for s in sorted(series, key=lambda x: x.get("date", "")):
            a, b = s["team_a"], s["team_b"]
            w = self._weight(s.get("date"))
            sa, sb = s["score"]
            maps = s.get("maps", [])
            known_a = sum(1 for mp in maps if mp["score"][0] > mp["score"][1])
            known_b = len(maps) - known_a
            # Map results with identity.
            outcomes: list[tuple[str | None, float, float]] = []
            for mp in maps:
                ra, rb = mp["score"]
                total = ra + rb
                # Round-share margin bonus: 13-2 is more informative than 14-12.
                margin = abs(ra - rb) / max(total, 1)
                outcomes.append((mp["map"], 1.0 if ra > rb else 0.0, 1.0 + margin))
            # Remaining maps without identity (only series score known).
            outcomes += [(None, 1.0, 1.0)] * max(0, sa - known_a)
            outcomes += [(None, 0.0, 1.0)] * max(0, sb - known_b)
            for m, res, mult in outcomes:
                edge = (self._base(a) + self._map_r(a, m)) - (self._base(b) + self._map_r(b, m))
                delta = k_map * w * mult * (res - expected(edge))
                if a in self.ratings:
                    self.ratings[a].form += delta
                    if m in self.ratings[a].map_ratings:
                        self.ratings[a].map_ratings[m] += K_MAP_SPECIFIC * w * (res - expected(edge))
                if b in self.ratings:
                    self.ratings[b].form -= delta
                    if m in self.ratings[b].map_ratings:
                        self.ratings[b].map_ratings[m] -= K_MAP_SPECIFIC * w * (res - expected(edge))
            winner = a if sa > sb else b
            label = f"{s.get('date', '?')} {s.get('event', s.get('match', source))}: {a} {sa}-{sb} {b}"
            for t in (a, b):
                if t in self.ratings:
                    self.ratings[t].series_log.append(label)
            self._add_h2h(a, b, w if winner == a else -w, label)

    # ------------------------------------------------------------------ public
    def h2h_adjustment(self, a: str, b: str) -> float:
        key = tuple(sorted((a, b)))
        net = self._h2h.get(key, 0.0)
        if key[0] != a:
            net = -net
        return max(-H2H_CAP, min(H2H_CAP, net * H2H_PER_SERIES))

    def h2h_records(self, a: str, b: str) -> list[str]:
        return list(self._h2h_records.get(tuple(sorted((a, b))), []))

    def home_bonus(self, team: str, home_region: str | None) -> float:
        if not home_region or team not in self.ratings:
            return 0.0
        if self.ratings[team].team.region == home_region:
            return float(self.ds.event.get("home_advantage_elo", 10))
        return 0.0

    def map_edge(self, a: str, b: str, m: str, home_region: str | None = None, use_h2h: bool = True) -> float:
        ea = self._base(a) + self._map_r(a, m) + self.home_bonus(a, home_region)
        eb = self._base(b) + self._map_r(b, m) + self.home_bonus(b, home_region)
        h = self.h2h_adjustment(a, b) if use_h2h else 0.0
        return ea - eb + h

    def map_win_prob(self, a: str, b: str, m: str, home_region: str | None = None, use_h2h: bool = True) -> float:
        return expected(self.map_edge(a, b, m, home_region, use_h2h))

    def power_rankings(self) -> list[TeamRating]:
        return sorted(self.ratings.values(), key=lambda r: r.elo, reverse=True)

    def best_maps(self, name: str) -> list[tuple[str, float]]:
        return sorted(self.ratings[name].map_ratings.items(), key=lambda kv: kv[1], reverse=True)


def logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))
