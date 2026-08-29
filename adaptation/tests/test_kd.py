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


# ===================== multi-teacher ensembling ==========================

from adaptation.kd import (  # noqa: E402
    KDDetectionLoss, fuse_teacher_probs, kd_class_term_from_target,
)

K = 4


def _tw(rows):
    return torch.tensor(rows, dtype=torch.float32)


def test_fusion_is_a_convex_combination():
    zs = [_logits(i) for i in range(K)]
    w = _tw([[0.4, 0.1, 0.0], [0.3, 0.2, 0.0], [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])
    p = fuse_teacher_probs(zs, w, temperature=2.0)
    assert p.shape == zs[0].shape
    assert (p >= 0).all() and (p <= 1).all()


def test_fusion_with_one_hot_weights_selects_that_teacher():
    """tau=0 / argmax routing must reduce exactly to the chosen teacher."""
    zs = [_logits(i) for i in range(K)]
    w = torch.zeros(K, NC); w[2, :] = 1.0
    p = fuse_teacher_probs(zs, w, temperature=2.0)
    assert torch.allclose(p, torch.sigmoid(zs[2] / 2.0), atol=1e-6)


def test_fusion_per_class_routing_is_independent_across_classes():
    """Different classes may select different teachers in the same forward."""
    zs = [_logits(i) for i in range(K)]
    w = torch.zeros(K, NC)
    w[0, 0] = 1.0; w[1, 1] = 1.0; w[3, 2] = 1.0
    p = fuse_teacher_probs(zs, w, temperature=2.0)
    assert torch.allclose(p[:, 0], torch.sigmoid(zs[0][:, 0] / 2.0), atol=1e-6)
    assert torch.allclose(p[:, 1], torch.sigmoid(zs[1][:, 1] / 2.0), atol=1e-6)
    assert torch.allclose(p[:, 2], torch.sigmoid(zs[3][:, 2] / 2.0), atol=1e-6)


def test_uniform_fusion_dilutes_a_lone_confident_teacher():
    """The predicted failure of competence-blind ensembling on a monopoly class.

    One teacher is confident, the rest output nothing. Uniform weighting drags
    the target toward zero; routing to the owner preserves it.
    """
    zs = [torch.full((1, NC, 2), -10.0) for _ in range(6)]
    zs[4] = zs[4].clone(); zs[4][0, 0, 0] = 10.0          # C5 knows class 0
    uniform = fuse_teacher_probs(zs, torch.full((6, NC), 1 / 6), temperature=1.0)
    w = torch.zeros(6, NC); w[4, :] = 1.0
    routed = fuse_teacher_probs(zs, w, temperature=1.0)
    assert routed[0, 0, 0] > 0.99
    assert uniform[0, 0, 0] < 0.20
    assert routed[0, 0, 0] > 4 * uniform[0, 0, 0]


def test_fusion_matches_single_teacher_path():
    """K=1 with weight 1.0 must reproduce kd_class_term exactly."""
    s, t = _logits(0), _logits(1)
    cw = torch.ones(NC) / NC
    kd_single, pc_single = kd_class_term(s, t, cw, temperature=2.0)
    p = fuse_teacher_probs([t], torch.ones(1, NC), temperature=2.0)
    kd_multi, pc_multi = kd_class_term_from_target(s, p, cw, temperature=2.0)
    assert torch.allclose(kd_single, kd_multi, atol=1e-6)
    assert torch.allclose(pc_single, pc_multi, atol=1e-6)


def test_identical_teachers_fuse_to_themselves():
    """Averaging teachers that agree changes nothing -- why unstable argmax is benign."""
    t = _logits(3)
    p = fuse_teacher_probs([t, t, t], torch.full((3, NC), 1 / 3), temperature=2.0)
    assert torch.allclose(p, torch.sigmoid(t / 2.0), atol=1e-6)


def test_unnormalized_columns_rejected():
    zs = [_logits(i) for i in range(2)]
    try:
        fuse_teacher_probs(zs, torch.full((2, NC), 0.9), temperature=2.0)
    except ValueError as e:
        assert "sum to 1" in str(e)
        return
    raise AssertionError("unnormalised teacher weights should raise")


def test_mismatched_teacher_shapes_rejected():
    try:
        fuse_teacher_probs([_logits(0), torch.randn(B, NC, A + 1)],
                           torch.full((2, NC), 0.5), temperature=2.0)
    except ValueError as e:
        assert "share arch" in str(e)
        return
    raise AssertionError("mismatched teacher shapes should raise")


def test_wrong_weight_shape_rejected():
    try:
        fuse_teacher_probs([_logits(0), _logits(1)], torch.ones(3, NC) / 3)
    except ValueError as e:
        assert "!=" in str(e)
        return
    raise AssertionError("wrong teacher_weights shape should raise")


def test_gradient_does_not_reach_teachers_through_fusion():
    s = _logits(0).requires_grad_(True)
    zs = [_logits(i + 1).requires_grad_(True) for i in range(K)]
    p = fuse_teacher_probs(zs, torch.full((K, NC), 1 / K), temperature=2.0)
    kd, _ = kd_class_term_from_target(s, p, torch.ones(NC) / NC)
    kd.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    for z in zs:
        assert z.grad is None or z.grad.abs().sum() == 0


def test_mask_uses_unsoftened_probabilities():
    """Why mask_probs is a separate argument from the target.

    Softening pulls every probability toward 0.5, so a confidence mask taken on
    the SOFTENED target stops separating objects from background. Here one anchor
    holds a real detection and three are background. At T=8 the background rises
    to sigmoid(-1.25)=0.223 and clears a 0.2 threshold, so masking on the target
    selects all four anchors; the unsoftened probabilities still isolate the one.
    """
    conf = 0.2
    t = torch.full((1, NC, 4), -10.0)
    t[0, 0, 0] = 10.0
    s = torch.full((1, NC, 4), 2.0)
    ones = torch.ones(1, NC)
    soft = fuse_teacher_probs([t], ones, temperature=8.0)   # the KD target
    hard = fuse_teacher_probs([t], ones, temperature=1.0)   # what the mask should use

    assert int((soft.amax(dim=1) > conf).sum()) == 4        # mask stops discriminating
    assert int((hard.amax(dim=1) > conf).sum()) == 1        # still selective

    kd_wrong, _ = kd_class_term_from_target(s, soft, torch.ones(NC), 8.0,
                                            mask_probs=None, teacher_conf=conf)
    kd_right, _ = kd_class_term_from_target(s, soft, torch.ones(NC), 8.0,
                                            mask_probs=hard, teacher_conf=conf)
    assert torch.isfinite(kd_wrong) and torch.isfinite(kd_right)
    assert not torch.allclose(kd_wrong, kd_right), (
        "masking on the softened target must differ from masking on real confidence")


def test_loss_defaults_to_uniform_weights():
    class _Dummy(torch.nn.Module):
        def forward(self, x):
            return x
    loss = KDDetectionLoss(base=None, teacher=[_Dummy(), _Dummy(), _Dummy()],
                           class_weights=[0.2, 0.3, 0.5])
    assert tuple(loss.tw.shape) == (3, NC)
    assert torch.allclose(loss.tw, torch.full((3, NC), 1 / 3))
    assert loss.teacher is loss.teachers[0]   # back-compat property


def test_loss_rejects_wrong_teacher_weight_shape():
    class _Dummy(torch.nn.Module):
        def forward(self, x):
            return x
    try:
        KDDetectionLoss(base=None, teacher=[_Dummy(), _Dummy()],
                        class_weights=[0.2, 0.3, 0.5],
                        teacher_weights=[[1.0, 0.0, 0.0]])
    except ValueError:
        return
    raise AssertionError("wrong teacher_weights shape should raise")


if __name__ == "__main__":
    _run_all()
