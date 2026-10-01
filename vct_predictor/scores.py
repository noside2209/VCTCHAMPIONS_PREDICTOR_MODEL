"""Round-level model that turns a map win probability into exact scorelines.

A map is played the way VALORANT plays it:
  * two halves of 12 rounds; team A starts on attack or defence (50/50, the side
    choice usually goes to the team that did not pick the map),
  * rounds 1 and 13 are pistol rounds,
  * round 2 / 14 is the anti-eco: the pistol winner gets +30 points of round win chance,
  * round 3 / 15 is the bonus round: the pistol winner (on a bonus buy against full
    guns) is 8 points worse than usual,
  * first to 13; at 12-12 overtime is played in pairs of rounds (one per side, win by two).

Team A's gun-round win rate r is drawn per map from a normal distribution around r0
(sd ROUND_SD, truncated at +/- Z_CAP) to capture form, momentum and economy swings. r0
is solved so that, with neutral pistols and no side data, the map win probability
matches the rating model.

A MapContext adds what is known about this matchup on this map:
  * pistol win chances for team A on attack and on defence (from pistol records),
  * how attack- or defence-sided the map is, and each team's side lean.
Without side data the context is neutral: pistols follow overall strength
(0.5 + PISTOL_SKILL * (r - 0.5)) and both sides are equal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

ROUND_SD = 0.10  # calibrated so ~20-30% of maps are blowouts (loser <= 5), ~2-4% are 13-2 or worse, ~10% go to OT
Z_CAP = 2.5      # no freak whole-map form flips beyond 2.5 sd
PISTOL_SKILL = 0.6  # pistols are more random than gun rounds: skill shows through at 60% strength
CONV_ANTI_ECO = 0.30  # pistol winner's boost in round 2 / 14
CONV_BONUS = -0.08    # pistol winner's penalty in round 3 / 15 (bonus buy vs full buy)
MAX_OT_PAIRS = 6  # 14-12 ... 19-17; longer OTs are lumped into the last bucket
_Z_NODES = [-Z_CAP + 2 * Z_CAP * i / 20 for i in range(21)]
_Z_WEIGHTS = [math.exp(-x * x / 2) for x in _Z_NODES]
_Z_WEIGHTS = [w / sum(_Z_WEIGHTS) for w in _Z_WEIGHTS]


@dataclass(frozen=True)
class MapContext:
    """Matchup-specific map info, from team A's point of view (None = neutral)."""
    pistol_atk: float | None = None  # A's chance to win a pistol round while attacking
    pistol_def: float | None = None  # A's chance to win a pistol round while defending
    atk_shift: float = 0.0           # added to A's gun-round win rate while attacking
    def_shift: float = 0.0           # added to A's gun-round win rate while defending

    def rounded(self) -> "MapContext":
        rd = lambda x: None if x is None else round(x, 3)  # noqa: E731
        return MapContext(rd(self.pistol_atk), rd(self.pistol_def), round(self.atk_shift, 3), round(self.def_shift, 3))

    def flipped(self) -> "MapContext":
        """The same context from team B's point of view."""
        f = lambda x: None if x is None else 1 - x  # noqa: E731
        return MapContext(f(self.pistol_def), f(self.pistol_atk), -self.def_shift, -self.atk_shift)


NEUTRAL = MapContext()


def _clamp(x: float, lo: float = 0.02, hi: float = 0.98) -> float:
    return min(hi, max(lo, x))


def neutral_pistol(r: float) -> float:
    return 0.5 + PISTOL_SKILL * (r - 0.5)


def _node_params(r: float, r0: float, ctx: MapContext) -> tuple[float, float, float, float]:
    """A's attack/defence gun-round and pistol win chances for one form draw r."""
    r_atk, r_def = _clamp(r + ctx.atk_shift), _clamp(r + ctx.def_shift)
    form = PISTOL_SKILL * (r - r0)
    p_atk = neutral_pistol(r) if ctx.pistol_atk is None else _clamp(ctx.pistol_atk + form)
    p_def = neutral_pistol(r) if ctx.pistol_def is None else _clamp(ctx.pistol_def + form)
    return r_atk, r_def, p_atk, p_def


def _struct_dist(r_atk: float, r_def: float, p_atk: float, p_def: float, a_starts_atk: bool) -> dict[tuple[int, int], float]:
    """Exact final-score distribution for fixed per-round chances (one starting side)."""
    states = {(0, 0, 0): 1.0}  # (A rounds, B rounds, pistol winner this half: 1 = A, -1 = B)
    dist: dict[tuple[int, int], float] = {}
    for n in range(1, 25):
        a_atk = a_starts_atk != (n > 12)
        k = (n - 1) % 12
        base = r_atk if a_atk else r_def
        nxt: dict[tuple[int, int, int], float] = {}
        for (a, b, pw), pr in states.items():
            if k == 0:
                p = p_atk if a_atk else p_def
            elif k == 1:
                p = _clamp(base + CONV_ANTI_ECO * pw)
            elif k == 2:
                p = _clamp(base + CONV_BONUS * pw)
            else:
                p = base
            for win, q in ((1, p), (0, 1 - p)):
                na, nb = a + win, b + 1 - win
                if na == 13 or nb == 13:
                    dist[(na, nb)] = dist.get((na, nb), 0.0) + pr * q
                else:
                    key = (na, nb, (1 if win else -1) if k == 0 else pw)
                    nxt[key] = nxt.get(key, 0.0) + pr * q
        states = nxt
    # Overtime: one round on each side per pair; win the pair 2-0 to take the map.
    both_a, both_b = r_atk * r_def, (1 - r_atk) * (1 - r_def)
    split = 1 - both_a - both_b
    carry = sum(states.values())
    for k in range(MAX_OT_PAIRS):
        if k == MAX_OT_PAIRS - 1:
            tot = both_a + both_b
            dist[(14 + k, 12 + k)] = dist.get((14 + k, 12 + k), 0.0) + carry * both_a / tot
            dist[(12 + k, 14 + k)] = dist.get((12 + k, 14 + k), 0.0) + carry * both_b / tot
        else:
            dist[(14 + k, 12 + k)] = carry * both_a
            dist[(12 + k, 14 + k)] = carry * both_b
            carry *= split
    return dist


def _both_starts(r_atk, r_def, p_atk, p_def) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for start in (True, False):
        for k, v in _struct_dist(r_atk, r_def, p_atk, p_def, start).items():
            out[k] = out.get(k, 0.0) + 0.5 * v
    return out


# Neutral map win chance as a function of a fixed r, on a fine grid (used to solve r0 quickly).
_GRID_STEP = 0.0025
_GRID: list[float] = []


def _neutral_win(r: float) -> float:
    if not _GRID:
        for i in range(int(round(1 / _GRID_STEP)) + 1):
            x = _clamp(i * _GRID_STEP)
            d = _both_starts(x, x, neutral_pistol(x), neutral_pistol(x))
            _GRID.append(sum(v for (a, b), v in d.items() if a > b))
    x = min(max(r, 0.0), 1.0) / _GRID_STEP
    i = min(int(x), len(_GRID) - 2)
    f = x - i
    return _GRID[i] * (1 - f) + _GRID[i + 1] * f


@lru_cache(maxsize=4096)
def _r0_for_p(p_rounded: float) -> float:
    lo, hi = 0.02, 0.98
    for _ in range(40):
        mid = (lo + hi) / 2
        w = sum(wt * _neutral_win(_clamp(mid + ROUND_SD * z)) for z, wt in zip(_Z_NODES, _Z_WEIGHTS))
        if w < p_rounded:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def r0_for_p(p_map: float) -> float:
    """Mean gun-round win rate that reproduces a map win probability (neutral pistols, no side data)."""
    return _r0_for_p(round(_clamp(p_map), 3))


@lru_cache(maxsize=8192)
def _dist(p_rounded: float, ctx: MapContext) -> tuple[tuple[tuple[int, int], float], ...]:
    r0 = _r0_for_p(p_rounded)
    out: dict[tuple[int, int], float] = {}
    for z, wt in zip(_Z_NODES, _Z_WEIGHTS):
        r = _clamp(r0 + ROUND_SD * z)
        for k, v in _both_starts(*_node_params(r, r0, ctx)).items():
            out[k] = out.get(k, 0.0) + wt * v
    return tuple(sorted(out.items()))


def scoreline_distribution(p_map: float, ctx: MapContext = NEUTRAL) -> dict[tuple[int, int], float]:
    """Distribution over final map scores (team A rounds, team B rounds)."""
    # Neutral maps share a cache across close probabilities; matchups with side data get their own.
    p = round(_clamp(p_map) * 500) / 500 if ctx == NEUTRAL else round(_clamp(p_map), 3)
    return dict(_dist(p, ctx.rounded()))


def map_win_prob(p_map: float, ctx: MapContext = NEUTRAL) -> float:
    """Map win chance after pistol/side information is applied (equals p_map when neutral)."""
    if ctx == NEUTRAL:
        return p_map
    return sum(v for (a, b), v in scoreline_distribution(p_map, ctx).items() if a > b)


def pistol_probs(p_map: float, ctx: MapContext = NEUTRAL) -> tuple[float, float]:
    """Team A's chance to win a pistol round on attack and on defence."""
    r0 = r0_for_p(p_map)
    return (neutral_pistol(r0) if ctx.pistol_atk is None else ctx.pistol_atk,
            neutral_pistol(r0) if ctx.pistol_def is None else ctx.pistol_def)


def most_likely_score(p_map: float, winner_is_a: bool | None = None, ctx: MapContext = NEUTRAL):
    """Most likely exact scoreline, optionally conditional on the winner."""
    items = scoreline_distribution(p_map, ctx).items()
    if winner_is_a is True:
        items = [(k, v) for k, v in items if k[0] > k[1]]
    elif winner_is_a is False:
        items = [(k, v) for k, v in items if k[1] > k[0]]
    return max(items, key=lambda kv: kv[1])


def expected_rounds(p_map: float, ctx: MapContext = NEUTRAL) -> tuple[float, float]:
    dist = scoreline_distribution(p_map, ctx)
    return (sum(a * v for (a, _), v in dist.items()), sum(b * v for (_, b), v in dist.items()))


def blowout_prob(p_map: float, ctx: MapContext = NEUTRAL) -> float:
    """Chance the loser finishes on 5 rounds or fewer."""
    return sum(v for (a, b), v in scoreline_distribution(p_map, ctx).items() if min(a, b) <= 5)


def overtime_prob(p_map: float, ctx: MapContext = NEUTRAL) -> float:
    return sum(v for (a, b), v in scoreline_distribution(p_map, ctx).items() if a >= 14 or b >= 14)


@dataclass
class MapPlay:
    score: tuple[int, int]
    half: tuple[int, int]            # score after round 12
    a_started_atk: bool
    pistols: list[tuple[int, bool, str]]  # (round, A won?, A's side "atk"/"def")


def play_map(p_map: float, rng, ctx: MapContext = NEUTRAL) -> MapPlay:
    """Play one map round by round with sides, pistols and pistol conversions."""
    r0 = r0_for_p(p_map)
    z = max(-Z_CAP, min(Z_CAP, rng.gauss(0, 1)))
    r_atk, r_def, p_atk, p_def = _node_params(_clamp(r0 + ROUND_SD * z), r0, ctx)
    start_atk = rng.random() < 0.5
    a = b = 0
    pw = 0
    half = (0, 0)
    pistols: list[tuple[int, bool, str]] = []
    n = 0
    while True:
        n += 1
        if n <= 24:
            a_atk = start_atk != (n > 12)
            k = (n - 1) % 12
            base = r_atk if a_atk else r_def
            if k == 0:
                p = p_atk if a_atk else p_def
            elif k == 1:
                p = _clamp(base + CONV_ANTI_ECO * pw)
            elif k == 2:
                p = _clamp(base + CONV_BONUS * pw)
            else:
                p = base
        else:  # overtime alternates sides every round
            a_atk = (n % 2 == 1) == start_atk
            p = r_atk if a_atk else r_def
            k = -1
        win = rng.random() < p
        a += win
        b += not win
        if k == 0:
            pw = 1 if win else -1
            pistols.append((n, win, "atk" if a_atk else "def"))
        if n == 12:
            half = (a, b)
        if max(a, b) >= 13 and abs(a - b) >= 2:
            return MapPlay((a, b), half if n > 12 else (a, b), start_atk, pistols)


def sample_score(p_map: float, rng, ctx: MapContext = NEUTRAL) -> tuple[int, int]:
    return play_map(p_map, rng, ctx).score
