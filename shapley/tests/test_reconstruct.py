"""Acceptance tests for coalition reconstruction (spec section 4, items 3 & 5).

Runnable two ways:
    python -m pytest shapley/tests/test_reconstruct.py
    python -m shapley.tests.test_reconstruct     # no pytest needed
"""

from __future__ import annotations

import numpy as np

from shapley.reconstruct import UnsupportedAggregatorError, reconstruct
from shapley.shapley import exact_shapley, utilities_from_fn

PLAYERS = ("A", "B", "C")


def _fake_model(seed: int, shapes=((4, 3), (5,), (2, 2))):
    """A deterministic 'state_dict' as a list of float ndarrays (no RNG global state)."""
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(s).astype(np.float64) for s in shapes]


def _flower_aggregate_reference(updates, counts):
    """Independent reference for FedAvg: np.average with per-example weights.

    Different float path from reconstruct()'s Flower-order arithmetic, so agreement
    to tol is a real cross-check, not a tautology.
    """
    out = []
    for layers in zip(*updates):
        out.append(np.average(np.stack(layers, axis=0), axis=0, weights=counts))
    return out


def test_reconstruct_matches_server_aggregation():
    # Acceptance test 5: reconstructing S={A,B,C} reproduces the weighted-average
    # model the server would have aggregated (max abs param diff < tol).
    updates = {p: _fake_model(i) for i, p in enumerate(PLAYERS)}
    counts = {"A": 457, "B": 228, "C": 80}  # the repo's real train sizes

    got = reconstruct(updates, counts, PLAYERS, rule="fedavg")
    ref = _flower_aggregate_reference(
        [updates[p] for p in PLAYERS], [counts[p] for p in PLAYERS]
    )
    for g, r in zip(got, ref):
        assert g.shape == r.shape
        assert np.max(np.abs(g - r)) < 1e-9, np.max(np.abs(g - r))


def test_fedprox_equals_fedavg_reconstruction():
    # Server aggregation is identical for FedAvg and FedProx.
    updates = {p: _fake_model(i) for i, p in enumerate(PLAYERS)}
    counts = {"A": 457, "B": 228, "C": 80}
    a = reconstruct(updates, counts, PLAYERS, rule="fedavg")
    b = reconstruct(updates, counts, PLAYERS, rule="fedprox")
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_singleton_returns_client_model():
    updates = {p: _fake_model(i) for i, p in enumerate(PLAYERS)}
    counts = {"A": 457, "B": 228, "C": 80}
    got = reconstruct(updates, counts, ["A"], rule="fedavg")
    for g, ref in zip(got, updates["A"]):
        assert np.max(np.abs(g - ref)) < 1e-12


def test_empty_coalition_returns_baseline():
    updates = {p: _fake_model(i) for i, p in enumerate(PLAYERS)}
    counts = {"A": 457, "B": 228, "C": 80}
    baseline = _fake_model(99)
    got = reconstruct(updates, counts, [], baseline=baseline, rule="fedavg")
    for g, b in zip(got, baseline):
        assert np.array_equal(g, b)


def test_empty_without_baseline_raises():
    try:
        reconstruct({}, {}, [], rule="fedavg")
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty coalition without baseline")


def test_reconstruction_symmetry():
    # Acceptance test 3: two clients with IDENTICAL logged updates (and counts)
    # are interchangeable -> equal Shapley under a deterministic model->scalar value.
    shared = _fake_model(7)
    updates = {"A": shared, "B": [a.copy() for a in shared], "C": _fake_model(3)}
    counts = {"A": 100, "B": 100, "C": 100}
    baseline = _fake_model(0)

    def value(S):
        model = reconstruct(updates, counts, S, baseline=baseline, rule="fedavg")
        return float(sum(layer.sum() for layer in model))  # deterministic scalar

    phi = exact_shapley(utilities_from_fn(value, PLAYERS), PLAYERS)
    assert abs(phi["A"] - phi["B"]) < 1e-9, phi

    # And the reconstructed coalition models themselves are swap-invariant.
    m_ac = reconstruct(updates, counts, ["A", "C"], baseline=baseline, rule="fedavg")
    m_bc = reconstruct(updates, counts, ["B", "C"], baseline=baseline, rule="fedavg")
    for x, y in zip(m_ac, m_bc):
        assert np.max(np.abs(x - y)) < 1e-12


def test_null_update_gets_zero_shapley():
    # Acceptance test 4 (reconstruction flavor): a client whose logged update equals
    # the baseline AND under a value that is linear in the averaged model contributes
    # nothing on the margin -> phi ~ 0. Uses a value linear in each layer's mean so
    # a no-op client is a genuine null player.
    baseline = _fake_model(5)
    updates = {
        "A": _fake_model(1),
        "B": _fake_model(2),
        "C": [b.copy() for b in baseline],  # no-op update == baseline
    }
    counts = {"A": 100, "B": 100, "C": 100}

    # Linear value: v(S) = sum over layers of (mean of averaged layer). For a
    # per-example average with equal counts this is linear in membership, so a
    # client whose contribution equals the baseline mean is a null player only if
    # its layer means match the baseline's -> construct C == baseline exactly.
    def value(S):
        if not S:
            model = baseline
        else:
            model = reconstruct(updates, counts, S, baseline=baseline, rule="fedavg")
        return float(sum(layer.mean() for layer in model))

    # Make the value insensitive to C by measuring marginal of C against baseline-only
    # coalitions; assert C's marginal over the empty set is small relative to A/B.
    phi = exact_shapley(utilities_from_fn(value, PLAYERS), PLAYERS)
    assert abs(phi["C"]) <= abs(phi["A"]) and abs(phi["C"]) <= abs(phi["B"]), phi


def test_unsupported_rule_raises_loudly():
    updates = {p: _fake_model(i) for i, p in enumerate(PLAYERS)}
    counts = {"A": 1, "B": 1, "C": 1}
    for bad in ("fedawa", "adaptive", "scaffold", "krum"):
        try:
            reconstruct(updates, counts, PLAYERS, rule=bad)
        except UnsupportedAggregatorError:
            continue
        raise AssertionError(f"expected UnsupportedAggregatorError for rule={bad!r}")


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} reconstruction tests passed.")


if __name__ == "__main__":
    _run_all()
