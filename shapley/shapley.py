"""Exact Shapley values for the N=3 client contribution game.

Players are the 3 clients. With only 8 coalitions we compute Shapley exactly
from a full utility table v: 2^{players} -> R. `exact_shapley` hard-codes the
N=3 weights (1/3 for |S| in {0,2}, 1/6 for |S|=1); `general_shapley` is the
documented factorial-weight fallback for arbitrary N (kept for testing/parity,
not used by the driver).

Utilities are keyed by `frozenset` of player labels. v(emptyset) is the utility
of the empty coalition (the pre-round global / random-init baseline).
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Callable, Dict, FrozenSet, Hashable, Iterable, List, Sequence


Player = Hashable
Utilities = Dict[FrozenSet[Player], float]


def coalitions(players: Sequence[Player]) -> List[FrozenSet[Player]]:
    """All 2**len(players) coalitions as frozensets, including the empty set."""
    out: List[FrozenSet[Player]] = []
    for r in range(len(players) + 1):
        for combo in combinations(players, r):
            out.append(frozenset(combo))
    return out


def _require_full_table(utilities: Utilities, players: Sequence[Player]) -> None:
    expected = coalitions(players)
    missing = [S for S in expected if S not in utilities]
    if missing:
        pretty = ", ".join("{" + ",".join(map(str, sorted(map(str, S)))) + "}" for S in missing)
        raise KeyError(
            f"Utility table is missing {len(missing)} coalition(s): {pretty}. "
            f"All {len(expected)} coalitions of {list(players)} must be present."
        )


def exact_shapley(utilities: Utilities, players: Sequence[Player] = ("A", "B", "C")) -> Dict[Player, float]:
    """Exact Shapley values for the 3-player game using the closed-form N=3 weights.

    phi_i = sum_{S not containing i} w(|S|) * ( v(S ∪ {i}) - v(S) )
    with w(0)=w(2)=1/3 and w(1)=1/6.

    Args:
        utilities: full table v(S) for every one of the 8 coalitions (frozenset -> float).
        players:   the 3 player labels, default ("A", "B", "C").

    Returns:
        {player: phi} dict. By construction sum(phi) == v(all) - v(emptyset)
        (efficiency), which the caller can assert.
    """
    if len(players) != 3:
        raise ValueError(
            f"exact_shapley is specialized to N=3 (got {len(players)} players: {list(players)}). "
            "Use general_shapley for other N."
        )
    _require_full_table(utilities, players)

    # w(|S|) for S drawn from the 2 players other than i: sizes 0,1,2.
    w = {0: 1.0 / 3.0, 1: 1.0 / 6.0, 2: 1.0 / 3.0}

    phi: Dict[Player, float] = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for S in coalitions(others):  # subsets of the other 2 players
            gain = utilities[S | {i}] - utilities[S]
            total += w[len(S)] * gain
        phi[i] = total
    return phi


def general_shapley(utilities: Utilities, players: Sequence[Player]) -> Dict[Player, float]:
    """Exact Shapley for arbitrary N via factorial marginal-contribution weights.

    w(|S|) = |S|! (N-|S|-1)! / N!, summed over all S not containing i.
    Documented fallback; the driver uses `exact_shapley` for N=3. For N=3 this
    returns identical values (covered by a parity test).
    """
    _require_full_table(utilities, players)
    n = len(players)
    phi: Dict[Player, float] = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for S in coalitions(others):
            s = len(S)
            weight = factorial(s) * factorial(n - s - 1) / factorial(n)
            total += weight * (utilities[S | {i}] - utilities[S])
        phi[i] = total
    return phi


def utilities_from_fn(
    value_fn: Callable[[FrozenSet[Player]], float],
    players: Sequence[Player],
) -> Utilities:
    """Build the full utility table by calling `value_fn(S)` on every coalition.

    Convenience for wiring an evaluator (evaluate.v) into the Shapley functions.
    """
    return {S: float(value_fn(S)) for S in coalitions(players)}
