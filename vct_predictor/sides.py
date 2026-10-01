"""Pistol rounds and attack/defence sides, per team per map.

Data lives in data/side_stats.json (fill it with `python -m vct_predictor sync-vlr-sides`):

    "teams": {"LOUD": {"Ascent": {"atk_pistol": [won, played], "def_pistol": [won, played],
                                  "atk_rounds": [won, played], "def_rounds": [won, played]}}}
    "map_atk_round_rate": {"Split": 0.53, ...}   # league-wide attack round win rate per map

Small samples are shrunk toward a prior: pistols toward what the team's overall strength
implies (PISTOL_PRIOR_WEIGHT pistols' worth), side rates toward the map average
(SIDE_PRIOR_WEIGHT rounds' worth). A team with no data on a map simply uses the prior.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .scores import NEUTRAL, MapContext, neutral_pistol, r0_for_p

if TYPE_CHECKING:
    from .ratings import RatingModel

PISTOL_PRIOR_WEIGHT = 10.0
SIDE_PRIOR_WEIGHT = 100.0
SIDES = ("atk", "def")


@dataclass
class SideRates:
    pistol: dict[str, float]          # shrunk pistol win rate by side
    pistol_raw: dict[str, tuple[int, int] | None]
    rounds: dict[str, float]          # shrunk gun/overall round win rate by side
    rounds_raw: dict[str, tuple[int, int] | None]
    has_data: bool


def log5(pa: float, pb: float) -> float:
    """Chance A beats B when A wins pa vs an average team and B wins pb vs an average team."""
    num = pa * (1 - pb)
    return num / (num + (1 - pa) * pb)


class SideModel:
    def __init__(self, data_dir: Path, model: "RatingModel"):
        path = Path(data_dir) / "side_stats.json"
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.teams: dict = raw.get("teams", {})
        self.map_atk: dict[str, float] = raw.get("map_atk_round_rate", {})
        self.model = model
        self.source = raw.get("source", "none")

    @property
    def has_any_data(self) -> bool:
        return any(self.teams.values())

    def map_atk_rate(self, m: str) -> float:
        return float(self.map_atk.get(m, 0.5))

    def _prior_pistol(self, team: str, m: str) -> float:
        """Pistol win rate vs an average Champions team implied by overall strength on this map."""
        field = [t for t in self.model.ratings if t != team]
        p_vs_avg = sum(self.model.elo_map_prob(team, o, m) for o in field) / len(field)
        return neutral_pistol(r0_for_p(p_vs_avg))

    def rates(self, team: str, m: str) -> SideRates:
        rec = self.teams.get(team, {}).get(m, {})
        prior = self._prior_pistol(team, m)
        atk = self.map_atk_rate(m)
        side_prior = {"atk": atk, "def": 1 - atk}
        pistol, pistol_raw, rounds, rounds_raw = {}, {}, {}, {}
        for s in SIDES:
            w, n = rec.get(f"{s}_pistol", [0, 0])
            pistol[s] = (w + PISTOL_PRIOR_WEIGHT * prior) / (n + PISTOL_PRIOR_WEIGHT)
            pistol_raw[s] = (w, n) if n else None
            w, n = rec.get(f"{s}_rounds", [0, 0])
            rounds[s] = (w + SIDE_PRIOR_WEIGHT * side_prior[s]) / (n + SIDE_PRIOR_WEIGHT)
            rounds_raw[s] = (w, n) if n else None
        return SideRates(pistol, pistol_raw, rounds, rounds_raw, bool(rec))

    def lean(self, team: str, m: str) -> float:
        """How attack-leaning a team is on a map, in round-win-rate points (positive = attack-sided).

        Only the imbalance between sides counts; overall strength is already in the rating."""
        r = self.rates(team, m)
        atk = self.map_atk_rate(m)
        return ((r.rounds["atk"] - atk) - (r.rounds["def"] - (1 - atk))) / 2

    def context(self, a: str, b: str, m: str) -> MapContext:
        """Pistol and side information for A vs B on map m, from A's point of view."""
        ta, tb = self.teams.get(a, {}).get(m), self.teams.get(b, {}).get(m)
        bias = self.map_atk_rate(m) - 0.5
        if not ta and not tb and bias == 0:
            return NEUTRAL
        ra, rb = self.rates(a, m), self.rates(b, m)
        has_pistols = any(x.pistol_raw[s] for x in (ra, rb) for s in SIDES)
        shift = bias + self.lean(a, m) + self.lean(b, m)
        return MapContext(
            pistol_atk=log5(ra.pistol["atk"], rb.pistol["def"]) if has_pistols else None,
            pistol_def=log5(ra.pistol["def"], rb.pistol["atk"]) if has_pistols else None,
            atk_shift=shift,
            def_shift=-shift,
        )
