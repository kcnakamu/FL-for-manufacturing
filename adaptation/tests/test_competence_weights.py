"""Unit tests for adaptation/competence_weights.py -- pure, no torch, no GPU."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adaptation.competence_weights import (  # noqa: E402
    SIGMA_FLOOR, _column_weights, derive_weights, load_competence, check_class_order,
)

CLASSES = ["Crazing", "Inclusion", "Patches", "Pitted_surface",
           "Rolled-in_scale", "Scratches"]
TEACHERS = [f"local_c{i}" for i in range(1, 7)]


def make_competence(mu_rows, sigma_rows=None, seeds=(0, 1, 2)) -> dict:
    """mu_rows[k][c] -> the aggregated-JSON shape derive_weights expects."""
    sigma_rows = sigma_rows or [[0.02] * len(CLASSES) for _ in mu_rows]
    return {
        "seeds": list(seeds),
        "class_names": list(CLASSES),
        "cells": {
            t: {c: {"mean": mu_rows[k][i], "std": sigma_rows[k][i]}
                for i, c in enumerate(CLASSES)}
            for k, t in enumerate(TEACHERS)
        },
    }


# The real measured matrix (rounded), used as a realistic fixture.
MEASURED_MU = [
    [0.298, 0.780, 0.923, 0.000, 0.402, 0.841],
    [0.270, 0.734, 0.938, 0.000, 0.000, 0.839],
    [0.029, 0.246, 0.040, 0.007, 0.406, 0.634],
    [0.165, 0.487, 0.030, 0.005, 0.013, 0.005],
    [0.000, 0.035, 0.249, 0.392, 0.000, 0.000],
    [0.260, 0.664, 0.870, 0.001, 0.002, 0.690],
]
MEASURED_SIGMA = [
    [0.042, 0.029, 0.005, 0.000, 0.039, 0.012],
    [0.043, 0.038, 0.016, 0.000, 0.000, 0.043],
    [0.014, 0.068, 0.023, 0.007, 0.012, 0.005],
    [0.020, 0.023, 0.013, 0.007, 0.005, 0.001],
    [0.000, 0.006, 0.067, 0.041, 0.000, 0.000],
    [0.014, 0.035, 0.023, 0.000, 0.000, 0.028],
]


def approx(a, b, tol=1e-9):
    assert abs(a - b) < tol, f"{a} != {b}"


# ---------------------------------------------------------------- columns ---

def test_columns_sum_to_one():
    cw = derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA), tau=3.0)
    for i in range(len(CLASSES)):
        approx(sum(cw.w[k][i] for k in range(6)), 1.0, 1e-12)


def test_tau_zero_is_argmax():
    w = _column_weights([0.1, 0.9, 0.3], [0.02] * 3, tau=0.0)
    assert w == [0.0, 1.0, 0.0]


def test_tau_zero_splits_exact_ties():
    w = _column_weights([0.5, 0.5, 0.1], [0.02] * 3, tau=0.0)
    assert w == [0.5, 0.5, 0.0]


def test_tau_inf_is_uniform():
    w = _column_weights([0.1, 0.9, 0.3], [0.02] * 3, tau=float("inf"))
    assert all(abs(v - 1 / 3) < 1e-12 for v in w)


def test_equal_competence_gives_equal_weight():
    w = _column_weights([0.4, 0.4, 0.4], [0.02] * 3, tau=3.0)
    assert all(abs(v - 1 / 3) < 1e-12 for v in w)


def test_weights_are_monotonic_in_competence():
    w = _column_weights([0.1, 0.5, 0.9], [0.02] * 3, tau=3.0)
    assert w[0] < w[1] < w[2]


def test_larger_tau_flattens():
    """The core property: more smoothing -> closer to uniform."""
    mu, sg = [0.30, 0.50, 0.70], [0.02] * 3
    spread = []
    for tau in (1.0, 3.0, 10.0):
        w = _column_weights(mu, sg, tau)
        spread.append(max(w) - min(w))
    assert spread[0] > spread[1] > spread[2]


def test_noise_scaling_flattens_when_sigma_is_large():
    """Identical means, different noise -> the noisy class gets flatter weights."""
    mu = [0.60, 0.65]
    tight = _column_weights(mu, [0.005, 0.005], tau=3.0)
    noisy = _column_weights(mu, [0.100, 0.100], tau=3.0)
    assert (max(tight) - min(tight)) > (max(noisy) - min(noisy))


def test_zero_sigma_does_not_divide_by_zero():
    w = _column_weights([0.0, 0.0, 0.4], [0.0, 0.0, 0.0], tau=3.0)
    approx(sum(w), 1.0, 1e-12)
    assert w[2] == max(w)
    # sigma_bar floors at SIGMA_FLOOR, so the gap is read as very significant
    assert w[2] > 0.99


def test_negative_tau_rejected():
    try:
        _column_weights([0.1, 0.2], [0.02, 0.02], tau=-1.0)
    except ValueError:
        return
    raise AssertionError("negative tau should raise")


# ------------------------------------------------------------- lambda_c -----

def test_lambda_tracks_best_teacher_and_normalizes():
    cw = derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA), tau=3.0)
    approx(sum(cw.lambda_c), 1.0, 1e-12)
    # Patches (best 0.938) must outrank Crazing (best 0.298).
    assert cw.lambda_c[CLASSES.index("Patches")] > cw.lambda_c[CLASSES.index("Crazing")]


def test_min_competence_gates_hopeless_classes():
    cw = derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA),
                        tau=3.0, min_competence=0.35)
    # Crazing's best teacher is 0.298 -> gated off entirely.
    approx(cw.lambda_c[CLASSES.index("Crazing")], 0.0)
    assert cw.lambda_c[CLASSES.index("Patches")] > 0


def test_all_classes_gated_raises():
    try:
        derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA), min_competence=0.99)
    except ValueError:
        return
    raise AssertionError("gating every class should raise, not return zeros")


# ------------------------------------------------- behaviour on real data ---

def test_monopoly_class_routes_to_its_owner():
    """C5 exclusively owns Pitted_surface -> weight must concentrate on it."""
    cw = derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA), tau=3.0)
    i = CLASSES.index("Pitted_surface")
    assert cw.w[4][i] > 0.99, cw.w[4][i]
    assert cw.effective_teachers(i) < 1.05


def test_contested_class_spreads_across_the_top_tier():
    """C1/C2 are statistically tied on Patches -> neither may dominate."""
    cw = derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA), tau=3.0)
    i = CLASSES.index("Patches")
    w1, w2 = cw.w[0][i], cw.w[1][i]
    assert 0.5 < w1 / w2 < 2.0, (w1, w2)
    assert cw.effective_teachers(i) > 2.0


def test_incompetent_teachers_get_negligible_weight():
    cw = derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA), tau=3.0)
    i = CLASSES.index("Pitted_surface")
    for k in (0, 1):                      # C1, C2 have exactly 0.000 pitted
        assert cw.w[k][i] < 1e-6


# ------------------------------------------------------------- loading -----

def test_single_seed_rejected(tmp_path=Path("/tmp")):
    p = tmp_path / "_one_seed.json"
    p.write_text(json.dumps(make_competence(MEASURED_MU, MEASURED_SIGMA, seeds=(0,))))
    try:
        load_competence(p)
    except ValueError as e:
        assert "seed" in str(e).lower()
        p.unlink()
        return
    raise AssertionError("single-seed input should be rejected")


def test_single_seed_override_allowed(tmp_path=Path("/tmp")):
    p = tmp_path / "_one_seed2.json"
    p.write_text(json.dumps(make_competence(MEASURED_MU, MEASURED_SIGMA, seeds=(0,))))
    d = load_competence(p, allow_single_seed=True)
    assert d["seeds"] == [0]
    p.unlink()


def test_wrong_file_shape_rejected(tmp_path=Path("/tmp")):
    """A single-run competence_matrix.json must not be mistaken for the aggregate."""
    p = tmp_path / "_wrong.json"
    p.write_text(json.dumps({"seed": 0, "class_names": CLASSES, "matrix": {}}))
    try:
        load_competence(p)
    except ValueError as e:
        assert "aggregate_competence" in str(e)
        p.unlink()
        return
    raise AssertionError("single-run matrix should be rejected")


def test_class_order_mismatch_is_caught(tmp_path=Path("/tmp")):
    cw = derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA))
    y = tmp_path / "_bad.yaml"
    shuffled = [CLASSES[1], CLASSES[0]] + CLASSES[2:]
    y.write_text("names:\n" + "".join(f"- {n}\n" for n in shuffled) + "nc: 6\n")
    try:
        check_class_order(cw, y)
    except ValueError as e:
        assert "Class order mismatch" in str(e)
        y.unlink()
        return
    raise AssertionError("shuffled class order should be caught")


def test_class_order_match_passes(tmp_path=Path("/tmp")):
    cw = derive_weights(make_competence(MEASURED_MU, MEASURED_SIGMA))
    y = tmp_path / "_good.yaml"
    y.write_text("names:\n" + "".join(f"- {n}\n" for n in CLASSES) + "nc: 6\n")
    check_class_order(cw, y)
    y.unlink()


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
