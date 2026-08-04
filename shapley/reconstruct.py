"""Coalition reconstruction — rebuild a coalition's aggregated model, retrain-free.

Clients in this repo send their FULL local model (state_dict as a list of numpy
arrays, in model._state_keys() order) each round, not a delta. FedAvg/FedProx
aggregate them identically on the server: a per-example-weighted average. So the
reconstructed model for a coalition S at round t is

    omega_S = sum_{i in S} (n_i / n_S) * omega_i^t          (n_S = sum_{i in S} n_i)

which this module computes by REPLICATING Flower's exact aggregation arithmetic
(and its operation order), so reconstructing S={A,B,C} reproduces the model the
server actually aggregated (acceptance test 5, max abs param diff < tol).

Aggregation-rule aware: FedAvg == FedProx here (identical server step). SCAFFOLD /
FedNova / robust rules (median/Krum/trimmed-mean) and this repo's own `fedawa` /
`adaptive` strategies do NOT use per-example weighting, so reconstruction under
them is DIFFERENT — this module refuses them loudly rather than silently applying
the wrong formula.
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, Hashable, Iterable, List, Optional, Sequence

import numpy as np

Player = Hashable
NDArrays = List[np.ndarray]


class UnsupportedAggregatorError(NotImplementedError):
    """Raised when reconstruction is requested under a non-weighted-average rule."""


def _weighted_average(updates: Sequence[NDArrays], counts: Sequence[int]) -> NDArrays:
    """Per-example weighted average, matching flwr.server.strategy.aggregate.aggregate.

    Replicates Flower's exact expression and operation order:
        weighted = [[layer * n_i for layer in w_i] ...]
        prime    = [reduce(np.add, layer_updates) / n_total for layer_updates in zip(*weighted)]
    so the float result is bit-comparable (to tol) with the server's aggregation.
    """
    n_total = sum(counts)
    if n_total <= 0:
        raise ValueError(f"Total example count must be positive, got {n_total}.")
    weighted = [[layer * n_i for layer in w] for w, n_i in zip(updates, counts)]
    # np.asarray keeps 0-d state entries (e.g. BatchNorm num_batches_tracked) as
    # ndarrays: a ufunc over 0-d inputs returns a numpy scalar, which set_parameters'
    # torch.from_numpy would reject. It is a no-op for the usual N-d layers.
    return [np.asarray(reduce(np.add, layer_updates) / n_total)
            for layer_updates in zip(*weighted)]


def _fedavg_reconstruct(
    updates_by_player: Dict[Player, NDArrays],
    counts_by_player: Dict[Player, int],
    subset: Iterable[Player],
    baseline: Optional[NDArrays],
) -> NDArrays:
    members = list(subset)
    if not members:
        if baseline is None:
            raise ValueError(
                "Empty coalition requested but no `baseline` (pre-round global / "
                "random-init) was provided for v(emptyset)."
            )
        return [np.array(a, copy=True) for a in baseline]

    missing = [p for p in members if p not in updates_by_player]
    if missing:
        raise KeyError(f"No logged update for player(s) {missing} in this round.")

    updates = [updates_by_player[p] for p in members]
    counts = [int(counts_by_player[p]) for p in members]

    n_layers = {len(u) for u in updates}
    if len(n_layers) != 1:
        raise ValueError(f"Players disagree on tensor count: {n_layers}. Same architecture required.")

    return _weighted_average(updates, counts)


# Rule -> reconstruct implementation. FedAvg and FedProx are the SAME server step.
AGGREGATORS = {
    "fedavg": _fedavg_reconstruct,
    "fedprox": _fedavg_reconstruct,  # proximal term is client-side only; server == FedAvg
}

# Registered strategies whose server aggregation is NOT per-example weighting, so
# the FedAvg reconstruction formula does not apply. Named explicitly so a run with
# one of these fails loudly instead of producing wrong Shapley values.
_KNOWN_NON_WEIGHTED = {
    "fedawa": "softmax over cosine-similarity scores (strategies/fedawa.py)",
    "adaptive": "adaptive precision/recall weighting (strategies/adaptive.py)",
}


def reconstruct(
    updates_by_player: Dict[Player, NDArrays],
    counts_by_player: Dict[Player, int],
    subset: Iterable[Player],
    baseline: Optional[NDArrays] = None,
    rule: str = "fedavg",
) -> NDArrays:
    """Reconstruct coalition `subset`'s aggregated model at a single round.

    Args:
        updates_by_player: player -> full model as list of ndarrays (state_dict order).
        counts_by_player:  player -> n_i (train image count) used as the FedAvg weight.
        subset:            the coalition S (iterable of players); empty -> `baseline`.
        baseline:          model for v(emptyset); required only if `subset` is empty.
        rule:              aggregation rule the run used. fedavg/fedprox supported.

    Returns:
        The aggregated model as a list of ndarrays, in the same order as the inputs.
    """
    rule = rule.lower()
    if rule not in AGGREGATORS:
        if rule in _KNOWN_NON_WEIGHTED:
            raise UnsupportedAggregatorError(
                f"Aggregation rule {rule!r} uses {_KNOWN_NON_WEIGHTED[rule]}, NOT per-example "
                "weighted averaging. Coalition reconstruction under it must replicate that "
                "strategy's own weighting — the FedAvg formula would give wrong Shapley values. "
                "Implement a dedicated reconstructor before running Shapley on this rule."
            )
        raise UnsupportedAggregatorError(
            f"Unknown/unsupported aggregation rule {rule!r}. Supported: {sorted(AGGREGATORS)}. "
            "If the server aggregation changed (SCAFFOLD/FedNova/median/Krum/trimmed-mean), the "
            "reconstruction must change to match — do not assume weighted averaging."
        )
    return AGGREGATORS[rule](updates_by_player, counts_by_player, subset, baseline)
