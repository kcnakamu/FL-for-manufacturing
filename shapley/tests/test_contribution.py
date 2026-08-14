"""Tests for the per-class contribution matrix math (no YOLO / torch).

Runnable:
    python -m pytest shapley/tests/test_contribution.py
    python -m shapley.tests.test_contribution
"""

from __future__ import annotations

import math

from shapley.contribution import (
    OVERALL,
    class_retention_weights,
    contribution_matrix,
    per_class_records,
    retention_matrix,
)
from shapley.shapley import coalitions, exact_shapley

PLAYERS = ["0", "1", "2"]
CLASSES = ["Inclusion", "Patches", "Scratches"]

# The persistence worked example as the overall table.
WORKED = {
    "": 0.33, "0": 0.55, "1": 0.50, "2": 0.40,
    "01": 0.75, "02": 0.65, "12": 0.60, "012": 0.82,
}


def _records(tau, overall=WORKED, per_class_shift=None):
    """Build persistence-style records: per-class tables are the overall table
    shifted by a per-class constant (keeps marginals — hence Shapley — intact)."""
    per_class_shift = per_class_shift or {c: 0.0 for c in CLASSES}
    recs = []
    for key, v in overall.items():
        recs.append({
            "tau": tau, "coalition": list(key), "mAP50": v,
            "per_class": {c: v + s for c, s in per_class_shift.items()},
        })
    return recs


def _phi_direct(util_by_key):
    util = {frozenset(k): v for k, v in util_by_key.items()}
    return exact_shapley(util, PLAYERS)


def test_per_class_records_rewrites_utility():
    recs = _records(0, per_class_shift={"Inclusion": 0.1, "Patches": 0.0, "Scratches": -0.1})
    inc = per_class_records(recs, "Inclusion")
    assert len(inc) == 8
    by_coalition = {frozenset(r["coalition"]): r["mAP50"] for r in inc}
    assert math.isclose(by_coalition[frozenset()], 0.43)
    assert math.isclose(by_coalition[frozenset("012")], 0.92)


def test_per_class_records_drops_missing_class():
    recs = _records(0)
    del recs[3]["per_class"]["Patches"]
    assert len(per_class_records(recs, "Patches")) == 7
    assert len(per_class_records(recs, "Inclusion")) == 8


def test_matrix_matches_exact_shapley_per_column():
    shift = {"Inclusion": 0.05, "Patches": -0.02, "Scratches": 0.0}
    matrix = contribution_matrix(_records(0, per_class_shift=shift), PLAYERS, CLASSES)
    phi_expected = _phi_direct(WORKED)
    # A constant shift of all 8 utilities leaves every marginal unchanged,
    # so each class column must equal the overall Shapley values.
    for col in [OVERALL] + CLASSES:
        for p in PLAYERS:
            assert math.isclose(matrix[0][col][p], phi_expected[p], abs_tol=1e-9), (col, p)


def test_efficiency_axiom_per_column():
    shift = {"Inclusion": 0.07, "Patches": 0.0, "Scratches": -0.03}
    matrix = contribution_matrix(_records(0, per_class_shift=shift), PLAYERS, CLASSES)
    for col, s in [(OVERALL, 0.0)] + list(shift.items()):
        total = sum(matrix[0][col].values())
        v_full = WORKED["012"] + s
        v_empty = WORKED[""] + s
        assert math.isclose(total, v_full - v_empty, abs_tol=1e-9), col


def test_incomplete_class_table_skips_that_class_only():
    recs = _records(0)
    del recs[5]["per_class"]["Scratches"]  # Scratches table incomplete at tau=0
    matrix = contribution_matrix(recs, PLAYERS, CLASSES)
    assert "Scratches" not in matrix.get(0, {})
    assert "Inclusion" in matrix[0] and OVERALL in matrix[0]


def test_retention_matrix_per_column():
    recs = _records(0)
    faded = {k: (WORKED[""] + 0.5 * (v - WORKED[""])) for k, v in WORKED.items()}
    recs += _records(10, overall=faded)
    matrix = contribution_matrix(recs, PLAYERS, CLASSES)
    rho = retention_matrix(matrix, ref_tau=0)
    for col in [OVERALL] + CLASSES:
        for p in PLAYERS:
            assert math.isclose(rho[0][col][p], 1.0, abs_tol=1e-9)
            assert math.isclose(rho[10][col][p], 0.5, abs_tol=1e-9), (col, p)


def _matrix_from_phis(phi0_by_class, phi_end_by_class=None):
    m = {0: {c: dict(phis) for c, phis in phi0_by_class.items()}}
    if phi_end_by_class is not None:
        m[10] = {c: dict(phis) for c, phis in phi_end_by_class.items()}
    return m


def test_weights_static_mode_orders_and_normalizes():
    phi0 = {
        "Inclusion": {"0": 0.30, "1": 0.10, "2": 0.01},  # offline contribute 0.40
        "Patches":   {"0": 0.05, "1": 0.05, "2": 0.02},  # offline contribute 0.10
        "Scratches": {"0": 0.00, "1": 0.00, "2": 0.30},  # offline contribute 0.00
    }
    w = class_retention_weights(_matrix_from_phis(phi0), ["0", "1"], CLASSES, mode="static")
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    assert w["Inclusion"] > w["Patches"] > w["Scratches"] == 0.0
    assert math.isclose(w["Inclusion"], 0.8, abs_tol=1e-9)


def test_weights_lost_mode_uses_the_drop():
    phi0 = {
        "Inclusion": {"0": 0.30, "1": 0.10, "2": 0.01},
        "Patches":   {"0": 0.20, "1": 0.00, "2": 0.02},
        "Scratches": {"0": 0.10, "1": 0.00, "2": 0.30},
    }
    phi_end = {
        "Inclusion": {"0": 0.10, "1": 0.00, "2": 0.05},  # lost 0.20 + 0.10 = 0.30
        "Patches":   {"0": 0.20, "1": 0.00, "2": 0.05},  # lost 0.00 (retained)
        "Scratches": {"0": 0.20, "1": 0.00, "2": 0.40},  # gained -> clipped to 0
    }
    w = class_retention_weights(_matrix_from_phis(phi0, phi_end), ["0", "1"],
                                CLASSES, mode="lost", tau_end=10)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(w["Inclusion"], 1.0, abs_tol=1e-9)
    assert w["Patches"] == 0.0 and w["Scratches"] == 0.0


def test_weights_persistent_mode_uses_tau_end_contribution():
    phi0 = {
        "Inclusion": {"0": 0.30, "1": 0.10, "2": 0.01},
        "Patches":   {"0": 0.20, "1": 0.00, "2": 0.02},
        "Scratches": {"0": 0.10, "1": 0.00, "2": 0.30},
    }
    phi_end = {
        "Inclusion": {"0": 0.40, "1": 0.20, "2": 0.05},  # offline persist 0.60
        "Patches":   {"0": 0.10, "1": 0.10, "2": 0.05},  # offline persist 0.20
        "Scratches": {"0": 0.00, "1": 0.00, "2": 0.40},  # offline persist 0.00
    }
    w = class_retention_weights(_matrix_from_phis(phi0, phi_end), ["0", "1"],
                                CLASSES, mode="persistent", tau_end=10)
    assert math.isclose(sum(w.values()), 1.0, abs_tol=1e-9)
    # Uses tau_end (not tau0): Inclusion where offline STILL matter -> highest;
    # Scratches (C absorbed it) -> zero. Opposite ranking to 'lost' here.
    assert w["Inclusion"] > w["Patches"] > w["Scratches"] == 0.0
    assert math.isclose(w["Inclusion"], 0.75, abs_tol=1e-9)   # 0.60 / 0.80


def test_weights_persistent_vs_lost_can_disagree():
    # Scratches: offline drop a lot (lost-mode high) but nothing persists
    # (persistent-mode zero) -- the E5/t*=1 situation.
    phi0 = {
        "Inclusion": {"0": 0.50, "1": 0.25, "2": 0.0},
        "Patches":   {"0": 0.43, "1": 0.34, "2": 0.0},
        "Scratches": {"0": 0.30, "1": 0.20, "2": 0.0},
    }
    phi_end = {
        "Inclusion": {"0": 0.46, "1": 0.33, "2": 0.0},   # persists
        "Patches":   {"0": 0.41, "1": 0.28, "2": 0.0},   # persists
        "Scratches": {"0": 0.16, "1": 0.10, "2": 0.0},   # halved (C absorbed)
    }
    m = _matrix_from_phis(phi0, phi_end)
    lost = class_retention_weights(m, ["0", "1"], CLASSES, mode="lost", tau_end=10)
    pers = class_retention_weights(m, ["0", "1"], CLASSES, mode="persistent", tau_end=10)
    assert lost["Scratches"] == max(lost.values())        # lost over-weights Scratches
    assert pers["Scratches"] == min(pers.values())        # persistent down-weights it
    assert pers["Inclusion"] == max(pers.values())


def test_weights_negative_phi_clipped():
    phi0 = {c: {"0": -0.2, "1": -0.1, "2": 0.3} for c in CLASSES}
    w = class_retention_weights(_matrix_from_phis(phi0), ["0", "1"], CLASSES, mode="static")
    # All offline contributions negative -> clipped to zero -> uniform fallback.
    for c in CLASSES:
        assert math.isclose(w[c], 1.0 / 3.0, abs_tol=1e-9)


def test_weights_all_zero_falls_back_to_uniform():
    phi0 = {c: {"0": 0.0, "1": 0.0, "2": 0.5} for c in CLASSES}
    w = class_retention_weights(_matrix_from_phis(phi0), ["0", "1"], CLASSES, mode="lost")
    for c in CLASSES:
        assert math.isclose(w[c], 1.0 / 3.0, abs_tol=1e-9)


def test_weights_bad_mode_raises():
    try:
        class_retention_weights({0: {}}, ["0"], CLASSES, mode="banana")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown mode")


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} contribution tests passed.")


if __name__ == "__main__":
    _run_all()
