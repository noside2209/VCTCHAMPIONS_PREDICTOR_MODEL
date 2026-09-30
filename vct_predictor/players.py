"""Player layer: roles, agent pools, per-map agent assignment and projected stat lines.

Projection model (per map):
  * Expected rounds for each side come from the round-level score model.
  * Team kills per round rise with the share of rounds a team wins
    (K = 3.35 + 1.6 * (round_share - 0.5)); a team's deaths are the opponent's kills,
    so both box scores always balance.
  * Kills, deaths, assists and first kills/deaths are shared out within a team by
    role weights x player impact (stars take a bigger share of kills and fewer deaths).
  * ACS, ADR, KAST and rating are derived from the per-round rates with
    formulas calibrated to typical VCT box scores.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path

from .data import Dataset
from .series import SeriesPrediction

KILL_W = {"Duelist": 1.15, "Flex": 1.03, "Initiator": 0.95, "Controller": 0.92, "Sentinel": 0.95}
DEATH_W = {"Duelist": 1.15, "Flex": 1.02, "Initiator": 0.98, "Controller": 0.95, "Sentinel": 0.90}
ASSIST_W = {"Duelist": 0.6, "Flex": 1.0, "Initiator": 1.45, "Controller": 1.2, "Sentinel": 0.8}
ENTRY_W = {"Duelist": 2.2, "Flex": 1.2, "Initiator": 0.8, "Controller": 0.6, "Sentinel": 0.7}
# Off-role agents a player commonly fills (e.g. sentinel players on Viper when a comp runs no sentinel).
SECONDARY = {"Sentinel": {"Viper", "Astra", "Chamber"}, "Duelist": {"Chamber"}, "Initiator": {"Viper", "Gekko"},
             "Controller": {"Viper", "Astra", "Clove", "Omen"}}


@dataclass
class PlayerInfo:
    name: str
    team: str
    role: str
    igl: bool
    agents: list[str]
    source: str
    impact: float
    vlr: dict | None = None  # per-round stats synced from VLR (kpr, dpr, apr, fkpr, fdpr, rating, ...)


@dataclass
class StatLine:
    player: str
    team: str
    role: str
    agent: str
    rounds: float
    kills: float
    deaths: float
    assists: float
    fk: float
    fd: float
    acs: float
    adr: float
    kast: float
    rating: float

    @property
    def kd(self) -> float:
        return self.kills / max(self.deaths, 1e-9)

    def as_list(self) -> list:
        return [self.player, self.agent, round(self.kills, 1), round(self.deaths, 1), round(self.assists, 1),
                round(self.acs), round(self.adr), round(100 * self.kast), round(self.fk, 1), round(self.fd, 1),
                round(self.rating, 2)]


class PlayerModel:
    def __init__(self, ds: Dataset, path: Path | str | None = None):
        raw = json.loads(Path(path or ds.data_dir / "players.json").read_text(encoding="utf-8"))
        self.map_comps: dict[str, list[str]] = raw["map_comps"]
        self.agent_roles: dict[str, str] = raw["agent_roles"]
        self.players: dict[str, list[PlayerInfo]] = {}
        for team_name, team in ds.teams.items():
            impacts = {p.name: p.impact for p in team.players}
            rows = raw["teams"].get(team_name)
            if not rows:
                raise KeyError(f"players.json has no entry for {team_name}")
            if sorted(r["name"] for r in rows) != sorted(impacts):
                raise ValueError(f"players.json roster for {team_name} does not match teams.json")
            self.players[team_name] = [
                PlayerInfo(r["name"], team_name, r["role"], r.get("igl", False), r["agents"],
                           r.get("source", "estimated"), impacts[r["name"]], r.get("vlr_stats")) for r in rows]

    # ------------------------------------------------------------ agents
    def assign_agents(self, team: str, map_name: str) -> dict[str, str]:
        return dict(self._assign(team, map_name))

    @lru_cache(maxsize=None)
    def _assign(self, team: str, map_name: str) -> tuple[tuple[str, str], ...]:
        comp = self.map_comps.get(map_name)
        roster = self.players[team]
        if not comp:  # unknown map: everyone on their main
            return tuple((p.name, p.agents[0]) for p in roster)
        best, best_cost = None, None
        for perm in permutations(range(len(comp))):
            cost = 0.0
            for p, slot in zip(roster, perm):
                agent = comp[slot]
                if agent in p.agents:
                    cost += p.agents.index(agent)
                else:
                    role = self.agent_roles.get(agent)
                    if role == p.role or p.role == "Flex":
                        cost += 6
                    elif agent in SECONDARY.get(p.role, ()):
                        cost += 7
                    else:
                        cost += 10
            if best_cost is None or cost < best_cost:
                best, best_cost = perm, cost
        assigned = {p.name: comp[slot] for p, slot in zip(roster, best)}
        # A player never shows up on an agent outside his pool when he has an agent of the
        # same role: e.g. a Neon/Waylay duelist replaces the comp's Jett with his own Neon.
        for p in roster:
            agent = assigned[p.name]
            if agent in p.agents:
                continue
            role = self.agent_roles.get(agent)
            taken = set(assigned.values())
            swap = next((a for a in p.agents if self.agent_roles.get(a) == role and a not in taken), None)
            if swap:
                assigned[p.name] = swap
        return tuple((p.name, assigned[p.name]) for p in roster)

    @staticmethod
    def base_rates(p: PlayerInfo, role: str) -> dict[str, float]:
        """Per-round rates used to share out team totals: real VLR numbers when synced, else role x impact."""
        est = {
            "kpr": 0.70 * KILL_W[role] * p.impact ** 2,
            "dpr": 0.68 * DEATH_W[role] / p.impact ** 0.7,
            "apr": 0.29 * ASSIST_W[role],
            "fkpr": 0.10 * ENTRY_W[role] * p.impact ** 1.5,
            "fdpr": 0.10 * ENTRY_W[role] ** 0.8 / p.impact,
        }
        if p.vlr and p.vlr.get("rounds", 0) >= 100:
            for k in est:
                if p.vlr.get(k):
                    est[k] = float(p.vlr[k])
        return est

    def effective_role(self, player: PlayerInfo, agent: str) -> str:
        """A flex player takes the role of the agent he is on for this map."""
        return self.agent_roles.get(agent, player.role) if player.role == "Flex" else player.role

    # ------------------------------------------------------------ stats
    def project_map(self, team: str, opp: str, map_name: str, rounds_for: float, rounds_against: float) -> list[StatLine]:
        rounds = rounds_for + rounds_against
        share = rounds_for / rounds
        team_k = 3.35 + 1.6 * (share - 0.5)
        team_d = 3.35 + 1.6 * (0.5 - share)
        team_a = 1.45 + 0.5 * (share - 0.5)
        team_fk = 0.5 + 0.6 * (share - 0.5)
        team_fd = 1.0 - team_fk
        roster = self.players[team]
        agents = self.assign_agents(team, map_name)
        roles = {p.name: self.effective_role(p, agents[p.name]) for p in roster}
        rates = {p.name: self.base_rates(p, roles[p.name]) for p in roster}
        norm = lambda key: {k: v[key] / sum(r[key] for r in rates.values()) for k, v in rates.items()}  # noqa: E731
        kw, dw, aw, ew, fdw = norm("kpr"), norm("dpr"), norm("apr"), norm("fkpr"), norm("fdpr")
        out = []
        for p in roster:
            kpr = team_k * kw[p.name]
            dpr = team_d * dw[p.name]
            apr = team_a * aw[p.name]
            fkpr = team_fk * ew[p.name]
            fdpr = team_fd * fdw[p.name]
            adr = 30 + 155 * kpr
            acs = 30 + 250 * kpr + 40 * fkpr
            kast = min(0.92, max(0.5, 0.55 + 0.25 * kpr + 0.22 * apr - 0.10 * dpr + 0.08 * (share - 0.5)))
            rating = 1.0 + 1.1 * (kpr - 0.68) - 0.9 * (dpr - 0.68) + 0.4 * (kast - 0.72) + 0.8 * (fkpr - fdpr)
            out.append(StatLine(p.name, team, roles[p.name], agents[p.name], rounds, kpr * rounds, dpr * rounds,
                                apr * rounds, fkpr * rounds, fdpr * rounds, acs, adr, kast, rating))
        return out

    def project_series(self, pred: SeriesPrediction) -> dict:
        """Per-map stat lines for both teams plus a series projection weighted by P(map is played)."""
        played = map_played_probs([m.p_a for m in pred.maps], pred.bo)
        per_map = []
        totals: dict[str, dict[str, float]] = {}
        for mp, p_play in zip(pred.maps, played):
            ra, rb = mp.expected_rounds
            lines_a = self.project_map(pred.team_a, pred.team_b, mp.map, ra, rb)
            lines_b = self.project_map(pred.team_b, pred.team_a, mp.map, rb, ra)
            per_map.append({"map": mp.map, "p_played": p_play, "a": lines_a, "b": lines_b})
            for ln in lines_a + lines_b:
                t = totals.setdefault(ln.player, {"team": ln.team, "role": ln.role, "rounds": 0.0, "kills": 0.0,
                                                  "deaths": 0.0, "assists": 0.0, "fk": 0.0, "fd": 0.0,
                                                  "acs_x": 0.0, "adr_x": 0.0, "kast_x": 0.0, "rating_x": 0.0,
                                                  "maps": 0.0})
                w = p_play
                t["maps"] += w
                t["rounds"] += w * ln.rounds
                for k in ("kills", "deaths", "assists", "fk", "fd"):
                    t[k] += w * getattr(ln, k)
                for k in ("acs", "adr", "kast", "rating"):
                    t[k + "_x"] += w * ln.rounds * getattr(ln, k)
        series = []
        for name, t in totals.items():
            r = t["rounds"]
            series.append(StatLine(name, t["team"], self._main_role(t["team"], name), "", r, t["kills"], t["deaths"],
                                   t["assists"], t["fk"], t["fd"], t["acs_x"] / r, t["adr_x"] / r,
                                   t["kast_x"] / r, t["rating_x"] / r))
        winner = pred.favourite
        mvp = max((s for s in series if s.team == winner), key=lambda s: s.rating)
        top_fragger = max(series, key=lambda s: s.kills)
        return {"maps": per_map, "series": series, "expected_maps": sum(played), "mvp": mvp, "top_fragger": top_fragger}

    def _main_role(self, team: str, name: str) -> str:
        return next(p.role for p in self.players[team] if p.name == name)

    def roster(self, team: str) -> list[PlayerInfo]:
        return self.players[team]


def map_played_probs(map_probs: list[float], bo: int) -> list[float]:
    """Probability that each map in the veto order actually gets played."""
    need = bo // 2 + 1
    states = {(0, 0): 1.0}
    out = []
    for pa in map_probs:
        out.append(sum(states.values()))
        nxt: dict[tuple[int, int], float] = {}
        for (a, b), p in states.items():
            for (na, nb), q in (((a + 1, b), pa), ((a, b + 1), 1 - pa)):
                if na < need and nb < need:
                    nxt[(na, nb)] = nxt.get((na, nb), 0.0) + p * q
        states = nxt
    return out
