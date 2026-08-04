"""Tests for the driver's pure assembly math (no YOLO / torch).

Verifies that (tau, coalition) -> mAP50 records become correct per-checkpoint
Shapley values and retention ratios, including the worked example and NaN
guarding on a near-zero reference contribution.

Runnable:
    python -m pytest shapley/tests/test_persistence.py
    python -m shapley.tests.test_persistence
"""

from __future__ import annotations

import math

from shapley.persistence import (
    coalition_label,
    retention_curve,
    shapley_over_checkpoints,
)

PLAYERS = ["0", "1", "2"]  # A, B, C


def _records_for(util_by_coalition, tau):
    """util_by_coalition: {'': v, '0': v, '01': v, '012': v, ...} -> records at one tau."""
    recs = []
    for key, v in util_by_coalition.items():
        recs.append({"tau": tau, "coalition": list(key), "mAP50": v})
    return recs


# The spec worked example, mapped onto player ids 0/1/2 (A/B/C).
WORKED = {
    "": 0.33, "0": 0.55, "1": 0.50, "2": 0.40,
    "01": 0.75, "02": 0.65, "12": 0.60, "012": 0.82,
}


def test_shapley_matches_worked_example():
    phi_by_tau, skipped = shapley_over_checkpoints(_records_for(WORKED, 0), PLAYERS)
    assert skipped == []
    phi = phi_by_tau[0]
    assert math.isclose(phi["0"], 0.230, abs_tol=1e-3), phi
    assert math.isclose(phi["1"], 0.180, abs_tol=1e-3), phi
    assert math.isclose(phi["2"], 0.080, abs_tol=1e-3), phi


def test_incomplete_checkpoint_is_skipped_not_miscomputed():
    recs = _records_for(WORKED, 0)
    # tau=10 is missing coalition '02' -> its table is incomplete.
    partial = {k: v for k, v in WORKED.items() if k != "02"}
    recs += _records_for(partial, 10)
    phi_by_tau, skipped = shapley_over_checkpoints(recs, PLAYERS)
    assert 0 in phi_by_tau and 10 not in phi_by_tau
    assert skipped == [(10, 1)]


def test_retention_is_one_at_reference_and_decays():
    # tau=0 uses WORKED; tau=10 halves every player's marginal by scaling utilities.
    recs = _records_for(WORKED, 0)
    faded = {k: (WORKED[""] + 0.5 * (v - WORKED[""])) for k, v in WORKED.items()}
    recs += _records_for(faded, 10)
    phi_by_tau, _ = shapley_over_checkpoints(recs, PLAYERS)
    rho = retention_curve(phi_by_tau, ref_tau=0)
    for p in PLAYERS:
        assert math.isclose(rho[0][p], 1.0, abs_tol=1e-9)
        assert math.isclose(rho[10][p], 0.5, abs_tol=1e-9), (p, rho[10][p])


def test_retention_nan_on_zero_reference():
    # Make player 2 a null player at tau=0 (phi=0) -> rho must be NaN, not a crash.
    null2 = {
        "": 0.20, "0": 0.50, "1": 0.45, "2": 0.20,
        "01": 0.65, "02": 0.50, "12": 0.45, "012": 0.65,
    }
    recs = _records_for(null2, 0) + _records_for(null2, 10)
    phi_by_tau, _ = shapley_over_checkpoints(recs, PLAYERS)
    assert math.isclose(phi_by_tau[0]["2"], 0.0, abs_tol=1e-12)
    rho = retention_curve(phi_by_tau, ref_tau=0)
    assert math.isnan(rho[10]["2"])
    assert not math.isnan(rho[10]["0"])  # non-null players still fine


def test_coalition_labels():
    assert coalition_label(frozenset(), PLAYERS) == "none"
    assert coalition_label(frozenset(["0"]), PLAYERS) == "A"
    assert coalition_label(frozenset(["0", "2"]), PLAYERS) == "AC"
    assert coalition_label(frozenset(["0", "1", "2"]), PLAYERS) == "ABC"


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} persistence tests passed.")


if __name__ == "__main__":
    _run_all()
