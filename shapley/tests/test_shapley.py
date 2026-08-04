"""Acceptance tests for exact N=3 Shapley (spec section 4, items 1-4).

Runnable two ways:
    python -m pytest shapley/tests/test_shapley.py
    python -m shapley.tests.test_shapley        # no pytest needed
"""

from __future__ import annotations

import math

from shapley.shapley import coalitions, exact_shapley, general_shapley, utilities_from_fn

# Fixed player labels used across the suite.
PLAYERS = ("A", "B", "C")


def _table(**by_str):
    """Build a utility table from string keys like ''/'A'/'AB'/'ABC'."""
    return {frozenset(k): float(v) for k, v in by_str.items()}


# The worked example from the spec (section 4.2).
WORKED = _table(**{
    "": 0.33, "A": 0.55, "B": 0.50, "C": 0.40,
    "AB": 0.75, "AC": 0.65, "BC": 0.60, "ABC": 0.82,
})


def test_worked_example():
    phi = exact_shapley(WORKED, PLAYERS)
    assert math.isclose(phi["A"], 0.230, abs_tol=1e-3), phi
    assert math.isclose(phi["B"], 0.180, abs_tol=1e-3), phi
    assert math.isclose(phi["C"], 0.080, abs_tol=1e-3), phi


def test_efficiency_worked():
    phi = exact_shapley(WORKED, PLAYERS)
    grand = WORKED[frozenset(PLAYERS)] - WORKED[frozenset()]
    assert math.isclose(sum(phi.values()), grand, abs_tol=1e-9)


def test_efficiency_many_tables():
    # Efficiency must hold for ANY utility table, not just the worked one.
    # Deterministic pseudo-values (no RNG) across a spread of tables.
    for seed in range(50):
        vals = {}
        for idx, S in enumerate(coalitions(PLAYERS)):
            # deterministic, varied, includes negatives (mAP-style utilities can be negative)
            vals[S] = math.sin(seed * 1.7 + idx * 0.9) - 0.5 * (len(S) == 1)
        phi = exact_shapley(vals, PLAYERS)
        grand = vals[frozenset(PLAYERS)] - vals[frozenset()]
        assert math.isclose(sum(phi.values()), grand, abs_tol=1e-9), (seed, phi)


def test_symmetry():
    # A and B are interchangeable in the utility table -> equal Shapley.
    sym = _table(**{
        "": 0.10, "A": 0.40, "B": 0.40, "C": 0.25,
        "AB": 0.55, "AC": 0.50, "BC": 0.50, "ABC": 0.70,
    })
    phi = exact_shapley(sym, PLAYERS)
    assert math.isclose(phi["A"], phi["B"], abs_tol=1e-12), phi


def test_null_player():
    # C never changes any coalition's utility -> phi_C == 0 exactly.
    v_base = {"": 0.20, "A": 0.50, "B": 0.45, "AB": 0.65}
    null = {}
    for S in coalitions(PLAYERS):
        key = "".join(sorted(p for p in S if p != "C"))
        null[S] = float(v_base[key])
    phi = exact_shapley(null, PLAYERS)
    assert math.isclose(phi["C"], 0.0, abs_tol=1e-12), phi
    # And the two contributing players still absorb the whole grand utility.
    grand = null[frozenset(PLAYERS)] - null[frozenset()]
    assert math.isclose(phi["A"] + phi["B"], grand, abs_tol=1e-12)


def test_general_matches_exact_for_n3():
    # The documented factorial-weight fallback must agree with the N=3 closed form.
    phi_exact = exact_shapley(WORKED, PLAYERS)
    phi_general = general_shapley(WORKED, PLAYERS)
    for p in PLAYERS:
        assert math.isclose(phi_exact[p], phi_general[p], abs_tol=1e-12), p


def test_utilities_from_fn_roundtrip():
    table = utilities_from_fn(lambda S: len(S), PLAYERS)
    assert table[frozenset()] == 0.0
    assert table[frozenset(PLAYERS)] == 3.0
    # equal marginal contribution -> equal, and efficiency holds
    phi = exact_shapley(table, PLAYERS)
    assert all(math.isclose(v, 1.0, abs_tol=1e-12) for v in phi.values()), phi


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} shapley tests passed.")


if __name__ == "__main__":
    _run_all()
