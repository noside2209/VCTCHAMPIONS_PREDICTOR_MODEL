"""Map veto simulation.

Team A starts the veto (in VCT this is usually the higher seed / coin-toss
winner). Each team:
  * bans the remaining map where its own win probability is lowest
    (i.e. the opponent's best map / its own permaban),
  * picks the remaining map where its own win probability is highest.

The decider is whatever map is left.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

# In simulations a team does not always make the "optimal" call: choices are sampled with
# weight exp(value / VETO_TEMPERATURE), so near-equal maps swap often and clear calls rarely change.
VETO_TEMPERATURE = 0.03


@dataclass
class VetoStep:
    team: str  # team name, or "decider"
    action: str  # "ban" | "pick" | "decider"
    map: str


def veto_sequence(bo: int, pool_size: int) -> list[str]:
    """Return the veto sequence for a series length and pool size.

    Uses the official VCT formats for a 7-map pool:
      Bo1: A ban, B ban, A ban, B ban, A ban, B ban, decider
      Bo3: A ban, B ban, A pick, B pick, A ban, B ban, decider
      Bo5: A ban, B ban, A pick, B pick, A pick, B pick, decider
    and generalises sensibly for other pool sizes.
    """
    if bo not in (1, 3, 5):
        raise ValueError("bo must be 1, 3 or 5")
    if pool_size < bo:
        raise ValueError(f"map pool of {pool_size} is too small for a Bo{bo}")
    turn = ["A", "B"]
    seq: list[str] = []
    remaining = pool_size
    picks_needed = bo - 1
    if bo == 1:
        i = 0
        while remaining > 1:
            seq.append(f"{turn[i % 2]}_ban")
            remaining -= 1
            i += 1
        return seq + ["decider"]
    # Opening bans (two if the pool allows it), then picks, then closing bans.
    opening_bans = min(2, remaining - bo)
    i = 0
    for _ in range(opening_bans):
        seq.append(f"{turn[i % 2]}_ban")
        remaining -= 1
        i += 1
    i = 0
    for _ in range(picks_needed):
        seq.append(f"{turn[i % 2]}_pick")
        remaining -= 1
        i += 1
    i = 0
    while remaining > 1:
        seq.append(f"{turn[i % 2]}_ban")
        remaining -= 1
        i += 1
    return seq + ["decider"]


def run_veto(
    team_a: str,
    team_b: str,
    pool: list[str],
    bo: int,
    p_a: Callable[[str], float],
    rng=None,
) -> tuple[list[VetoStep], list[tuple[str, str]]]:
    """Simulate the veto.

    ``p_a(map)`` returns team A's win probability on that map. With ``rng`` the calls are
    sampled (see VETO_TEMPERATURE) instead of always taking the best option.
    Returns (veto steps, maps to be played as [(map, picked_by)]).
    """
    remaining = list(pool)
    steps: list[VetoStep] = []
    played: list[tuple[str, str]] = []
    probs = {m: p_a(m) for m in pool}
    for action in veto_sequence(bo, len(pool)):
        if action == "decider":
            m = remaining.pop()
            steps.append(VetoStep("decider", "decider", m))
            played.append((m, "decider"))
            break
        side, kind = action.split("_")
        team = team_a if side == "A" else team_b
        own = (lambda m: probs[m]) if side == "A" else (lambda m: 1.0 - probs[m])
        if rng is not None:
            sign = -1.0 if kind == "ban" else 1.0
            vals = [sign * own(x) for x in remaining]
            top = max(vals)
            m = rng.choices(remaining, [math.exp((v - top) / VETO_TEMPERATURE) for v in vals])[0]
        # Stable tie-break by pool order.
        elif kind == "ban":
            m = min(remaining, key=lambda x: (own(x), pool.index(x)))
        else:
            m = max(remaining, key=lambda x: (own(x), -pool.index(x)))
        remaining.remove(m)
        steps.append(VetoStep(team, kind, m))
        if kind == "pick":
            played.append((m, team))
    return steps, played
