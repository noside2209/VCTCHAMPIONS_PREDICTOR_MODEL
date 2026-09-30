"""Round-level model that turns a map win probability into exact scorelines.

Each round is won by team A with probability r. To capture momentum/eco swings
(real maps have more blowouts than an i.i.d. coin would produce), r itself is
drawn per map from a normal distribution around r0 (sd = ROUND_SD). r0 is solved
so that the resulting map win probability matches the rating model.

First to 13; at 12-12 overtime is played in pairs of rounds (win by two).
"""
from __future__ import annotations

import math
from functools import lru_cache

ROUND_SD = 0.06
MAX_OT_PAIRS = 6  # 14-12 ... 19-17; longer OTs are lumped into the last bucket
# Discretised standard normal (17 nodes on [-3, 3]) used to integrate over r.
_Z_NODES = [-3.0 + 6.0 * i / 16 for i in range(17)]
_Z_WEIGHTS = [math.exp(-x * x / 2) for x in _Z_NODES]
_Z_WEIGHTS = [w / sum(_Z_WEIGHTS) for w in _Z_WEIGHTS]


def _regulation_and_ot(r: float) -> dict[tuple[int, int], float]:
    """Exact scoreline distribution for a fixed per-round win probability r."""
    q = 1.0 - r
    # P(reach state (i, j)) for i, j <= 12 before anyone hits 13.
    reach = [[0.0] * 14 for _ in range(14)]
    reach[0][0] = 1.0
    dist: dict[tuple[int, int], float] = {}
    for total in range(0, 25):
        for i in range(0, 13):
            j = total - i
            if j < 0 or j > 12:
                continue
            p = reach[i][j]
            if p == 0.0:
                continue
            if i == 12 and j == 12:
                continue
            # A wins the round
            if i + 1 == 13:
                dist[(13, j)] = dist.get((13, j), 0.0) + p * r
            else:
                reach[i + 1][j] += p * r
            if j + 1 == 13:
                dist[(i, 13)] = dist.get((i, 13), 0.0) + p * q
            else:
                reach[i][j + 1] += p * q
    p_ot = reach[12][12]
    both_a, both_b, split = r * r, q * q, 2 * r * q
    carry = p_ot
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


def _mixture(r0: float) -> dict[tuple[int, int], float]:
    out: dict[tuple[int, int], float] = {}
    for x, w in zip(_Z_NODES, _Z_WEIGHTS):
        r = min(max(r0 + ROUND_SD * x, 0.02), 0.98)
        for k, v in _regulation_and_ot(r).items():
            out[k] = out.get(k, 0.0) + w * v
    return out


def _win_prob(dist: dict[tuple[int, int], float]) -> float:
    return sum(v for (a, b), v in dist.items() if a > b)


@lru_cache(maxsize=4096)
def _dist_for_p(p_rounded: float) -> tuple[tuple[tuple[int, int], float], ...]:
    lo, hi = 0.02, 0.98
    for _ in range(40):
        mid = (lo + hi) / 2
        if _win_prob(_mixture(mid)) < p_rounded:
            lo = mid
        else:
            hi = mid
    dist = _mixture((lo + hi) / 2)
    return tuple(sorted(dist.items()))


def scoreline_distribution(p_map: float) -> dict[tuple[int, int], float]:
    """Distribution over final map scores (team A rounds, team B rounds)."""
    return dict(_dist_for_p(round(min(max(p_map, 0.02), 0.98), 3)))


def most_likely_score(p_map: float, winner_is_a: bool | None = None) -> tuple[tuple[int, int], float]:
    """Most likely exact scoreline, optionally conditional on the winner."""
    dist = scoreline_distribution(p_map)
    items = dist.items()
    if winner_is_a is True:
        items = [(k, v) for k, v in items if k[0] > k[1]]
    elif winner_is_a is False:
        items = [(k, v) for k, v in items if k[1] > k[0]]
    k, v = max(items, key=lambda kv: kv[1])
    return k, v


def expected_rounds(p_map: float) -> tuple[float, float]:
    dist = scoreline_distribution(p_map)
    return (sum(a * v for (a, _), v in dist.items()), sum(b * v for (_, b), v in dist.items()))


def overtime_prob(p_map: float) -> float:
    return sum(v for (a, b), v in scoreline_distribution(p_map).items() if a >= 14 or b >= 14)
