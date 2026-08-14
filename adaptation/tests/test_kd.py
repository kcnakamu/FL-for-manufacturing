"""Tests for the KD math (CPU torch only, no YOLO training).

Runnable:
    python -m pytest adaptation/tests/test_kd.py
    python -m adaptation.tests.test_kd
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import torch

from adaptation.kd import kd_class_term, load_class_weights

B, NC, A = 2, 3, 20
CLASSES = ["Inclusion", "Patches", "Scratches"]


def _logits(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, NC, A, generator=g)


def test_zero_weights_give_zero_loss():
    kd, per_class = kd_class_term(_logits(0), _logits(1), torch.zeros(NC))
    assert kd.item() == 0.0
    assert per_class.shape == (NC,)
    assert (per_class > 0).all()  # per-class means themselves are nonzero


def test_matching_teacher_minimizes_the_term():
    t = _logits(1)
    w = torch.ones(NC) / NC
    kd_match, _ = kd_class_term(t.clone(), t, w)
    kd_off, _ = kd_class_term(t + 2.0, t, w)
    kd_off2, _ = kd_class_term(t - 2.0, t, w)
    assert kd_match < kd_off and kd_match < kd_off2
    # BCE(student==teacher) is the soft-target entropy: positive, not zero.
    assert kd_match.item() > 0.0


def test_weight_scaling_is_linear():
    s, t = _logits(0), _logits(1)
    w = torch.tensor([1.0, 0.0, 0.0])
    kd1, per_class = kd_class_term(s, t, w)
    kd2, _ = kd_class_term(s, t, 2.0 * w)
    assert math.isclose(kd2.item(), 2.0 * kd1.item(), rel_tol=1e-6)
    assert math.isclose(kd1.item(), per_class[0].item(), rel_tol=1e-6)


def test_only_weighted_classes_contribute():
    s, t = _logits(0), _logits(1)
    _, per_class = kd_class_term(s, t, torch.ones(NC))
    kd_c0, _ = kd_class_term(s, t, torch.tensor([1.0, 0.0, 0.0]))
    assert math.isclose(kd_c0.item(), per_class[0].item(), rel_tol=1e-6)


def test_gradient_flows_to_student_not_teacher():
    s = _logits(0).requires_grad_(True)
    t = _logits(1).requires_grad_(True)
    kd, _ = kd_class_term(s, t, torch.ones(NC) / NC)
    kd.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    assert t.grad is None or t.grad.abs().sum() == 0  # teacher detached inside


def test_teacher_conf_mask_drops_background_anchors():
    # Teacher very confident on anchor 0 only; all other anchors near-zero prob.
    # Student logits must be nonzero: BCE at logit 0 is ~log2 for ANY target,
    # which would make masked and unmasked means coincide.
    t = torch.full((1, NC, 4), -10.0)
    t[0, 0, 0] = 10.0
    s = torch.full((1, NC, 4), 2.0)
    w = torch.ones(NC)
    kd_all, _ = kd_class_term(s, t, w, teacher_conf=0.0)
    kd_masked, per_class_masked = kd_class_term(s, t, w, teacher_conf=0.5)
    # Masked version averages over the single confident anchor only.
    assert not math.isclose(kd_all.item(), kd_masked.item(), rel_tol=1e-3)
    T = 2.0
    p = torch.sigmoid(t[0, :, 0] / T)
    expected = (torch.nn.functional.binary_cross_entropy_with_logits(
        s[0, :, 0] / T, p, reduction="none") * T * T)
    assert torch.allclose(per_class_masked, expected, rtol=1e-5)


def test_temperature_softens_targets():
    t = torch.full((1, NC, 4), 4.0)
    s = torch.zeros(1, NC, 4)
    w = torch.ones(NC) / NC
    # Higher T -> softer target (closer to 0.5) but T^2 scaling: just check both run
    # and give different, finite values.
    kd_t1, _ = kd_class_term(s, t, w, temperature=1.0)
    kd_t4, _ = kd_class_term(s, t, w, temperature=4.0)
    assert kd_t1.isfinite() and kd_t4.isfinite()
    assert not math.isclose(kd_t1.item(), kd_t4.item(), rel_tol=1e-3)


def test_shape_mismatch_raises():
    try:
        kd_class_term(torch.zeros(1, NC, 4), torch.zeros(1, NC, 5), torch.ones(NC))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on shape mismatch")


def _weights_file(payload) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(payload, tmp)
    tmp.close()
    return tmp.name


def test_load_class_weights_roundtrip():
    # Same wrapper shape as shapley/contribution.py writes.
    path = _weights_file({"weights": {"Inclusion": 0.6, "Patches": 0.4, "Scratches": 0.0},
                          "mode": "lost", "offline": ["0", "1"]})
    try:
        w = load_class_weights(path, CLASSES)
        assert w == [0.6, 0.4, 0.0]
    finally:
        Path(path).unlink()


def test_load_class_weights_renormalizes_and_validates():
    path = _weights_file({"weights": {"Inclusion": 2.0, "Patches": 1.0, "Scratches": 1.0}})
    try:
        w = load_class_weights(path, CLASSES)
        assert math.isclose(sum(w), 1.0, abs_tol=1e-9) and w[0] == 0.5
    finally:
        Path(path).unlink()

    missing = _weights_file({"weights": {"Inclusion": 1.0}})
    try:
        load_class_weights(missing, CLASSES)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for missing classes")
    finally:
        Path(missing).unlink()


def test_load_class_weights_all_zero_uniform():
    path = _weights_file({"weights": {c: 0.0 for c in CLASSES}})
    try:
        w = load_class_weights(path, CLASSES)
        assert all(math.isclose(v, 1 / 3, abs_tol=1e-9) for v in w)
    finally:
        Path(path).unlink()


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} kd tests passed.")


if __name__ == "__main__":
    _run_all()
