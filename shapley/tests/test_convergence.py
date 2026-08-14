"""Tests for the disruption-timing plateau math (no YOLO / torch).

Runnable:
    python -m pytest shapley/tests/test_convergence.py
    python -m shapley.tests.test_convergence
"""

from __future__ import annotations

from shapley.convergence import convergence_round, recommend_t_star


def test_monotone_improvement_never_converges():
    rounds = list(range(1, 8))
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert convergence_round(rounds, values, min_delta=0.005, patience=2) is None


def test_step_then_flat_converges_at_step():
    rounds = [1, 2, 3, 4, 5, 6]
    values = [0.2, 0.5, 0.72, 0.72, 0.72, 0.72]
    assert convergence_round(rounds, values, min_delta=0.005, patience=2) == 3


def test_noise_below_min_delta_is_a_plateau():
    # Wiggles of +-0.003 never beat best by > min_delta=0.005.
    rounds = [1, 2, 3, 4, 5]
    values = [0.70, 0.703, 0.699, 0.702, 0.701]
    assert convergence_round(rounds, values, min_delta=0.005, patience=2) == 1


def test_patience_boundary():
    rounds = [1, 2, 3]
    values = [0.7, 0.7, 0.7]
    # One flat round is not enough at patience=2; two are.
    assert convergence_round(rounds[:2], values[:2], patience=2) is None
    assert convergence_round(rounds, values, patience=2) == 1
    assert convergence_round(rounds[:2], values[:2], patience=1) == 1


def test_late_improvement_resets_patience():
    rounds = [1, 2, 3, 4, 5, 6]
    values = [0.5, 0.5, 0.6, 0.6, 0.6, 0.6]  # flat, jump at r3, flat again
    assert convergence_round(rounds, values, min_delta=0.005, patience=3) == 3


def test_round_ids_are_returned_not_indices():
    rounds = [4, 6, 8, 10]  # non-contiguous round ids
    values = [0.6, 0.65, 0.65, 0.65]
    assert convergence_round(rounds, values, min_delta=0.005, patience=2) == 6


def test_recommend_t_star_converged():
    rec = recommend_t_star([1, 2, 3, 4], [0.3, 0.6, 0.6, 0.6])
    assert rec["converged"] is True
    assert rec["t_star"] == 2
    assert rec["mAP50_at_t_star"] == 0.6


def test_recommend_t_star_still_improving():
    rec = recommend_t_star([1, 2, 3], [0.3, 0.4, 0.5])
    assert rec["converged"] is False
    assert rec["t_star"] == 3  # falls back to last round, flagged in note
    assert "no plateau" in rec["note"]


def test_length_mismatch_raises():
    try:
        convergence_round([1, 2], [0.5])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on length mismatch")


def test_empty_curve():
    assert convergence_round([], []) is None


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} convergence tests passed.")


if __name__ == "__main__":
    _run_all()
