"""Version-drift tripwire for the pinned Ultralytics API surface KD relies on.

adaptation/kd.py injects a criterion into DetectionModel and reuses
v8DetectionLoss internals. These assertions fail loudly if an Ultralytics
upgrade changes that surface (fallback then: pseudo-label distillation, see
adaptation/kd.py docstring). Imports ultralytics + builds a tiny model on CPU;
no training.

Runnable:
    python -m pytest adaptation/tests/test_ultralytics_api.py
    python -m adaptation.tests.test_ultralytics_api
"""

from __future__ import annotations

import inspect

import torch


def test_v8detectionloss_surface():
    from ultralytics.utils.loss import v8DetectionLoss

    assert hasattr(v8DetectionLoss, "parse_output")
    assert hasattr(v8DetectionLoss, "get_assigned_targets_and_loss")
    # __call__(preds, batch) -> (loss * batch_size, loss_detached)
    sig = inspect.signature(v8DetectionLoss.__call__)
    assert list(sig.parameters)[:3] == ["self", "preds", "batch"]


def test_basemodel_loss_respects_preset_criterion():
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolov8n.yaml", nc=3, verbose=False)
    sentinel = object()
    calls = {}

    def fake_criterion(preds, batch):
        calls["preds"] = preds
        return sentinel

    model.criterion = fake_criterion
    out = model.loss({"img": torch.zeros(1, 3, 64, 64)})
    assert out is sentinel, "DetectionModel.loss no longer honors a pre-set criterion"
    # Training-mode forward feeds the criterion the raw preds dict.
    assert isinstance(calls["preds"], dict) and "scores" in calls["preds"]


def test_detect_eval_returns_logits_dict():
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel("yolov8n.yaml", nc=3, verbose=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert isinstance(out, tuple) and len(out) == 2, \
        "eval-mode Detect no longer returns (y, preds)"
    preds = out[1]
    assert isinstance(preds, dict) and {"boxes", "scores", "feats"} <= set(preds)
    b, nc, a = preds["scores"].shape
    assert (b, nc) == (1, 3), "scores are no longer (batch, nc, anchors)"
    # Training mode returns the same dict directly -> shapes anchor-aligned.
    model.train()
    tr = model(torch.zeros(1, 3, 64, 64))
    assert isinstance(tr, dict) and tr["scores"].shape == preds["scores"].shape


def test_trainer_hooks_exist():
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model  # noqa: F401

    # KDDetectionTrainer overrides the no-arg _setup_train.
    sig = inspect.signature(BaseTrainer._setup_train)
    assert list(sig.parameters) == ["self"], "_setup_train signature changed"
    # label_loss_items is generic over self.loss_names (4th kd item works).
    src = inspect.getsource(DetectionTrainer.label_loss_items)
    assert "loss_names" in src


def test_kd_criterion_composes_with_real_loss():
    """End-to-end on CPU: KDDetectionLoss over a real tiny model + fake batch."""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils import IterableSimpleNamespace

    from adaptation.kd import KDDetectionLoss

    student = DetectionModel("yolov8n.yaml", nc=3, verbose=False)
    # v8DetectionLoss reads hyperparameters from model.args.
    student.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    teacher = DetectionModel("yolov8n.yaml", nc=3, verbose=False).eval()
    teacher.requires_grad_(False)

    criterion = KDDetectionLoss(v8DetectionLoss(student), teacher,
                                [0.5, 0.3, 0.2], lam=1.0, temperature=2.0,
                                student=student)
    student.criterion = criterion
    student.train()

    batch = {
        "img": torch.zeros(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }
    # Training forward: the backpropped loss (1st return) carries KD as a 4th
    # element (so loss.sum() trains on it), but the TRACKED vector (2nd return)
    # stays 3-wide -- Ultralytics sizes the validator's accumulator from it and
    # validates the KD-free EMA, so a 4-wide tracked vector crashes validation.
    loss, loss_items = student.loss(batch)
    assert loss.shape == (4,), f"expected 4-wide backprop loss (…, kd), got {loss.shape}"
    assert loss_items.shape == (3,), f"tracked loss must stay 3-wide, got {loss_items.shape}"
    assert torch.isfinite(loss).all()
    assert loss[3] > 0, "kd term should be positive for differing student/teacher"
    assert criterion.last_kd > 0  # logged per epoch via callback

    # Validation path (no_grad): plain 3-item detection loss, no teacher forward.
    with torch.no_grad():
        v_loss, v_items = student.loss(batch)
    assert v_loss.shape == (3,), f"expected 3-wide loss under no_grad, got {v_loss.shape}"
    assert v_items.shape == (3,)

    # Eval-mode path with grad enabled (belt-and-suspenders for validator paths
    # that don't disable grad): the eval() flag must also gate KD off.
    student.eval()
    e_loss, e_items = student.loss(batch)
    assert e_loss.shape == (3,), f"expected 3-wide loss in eval mode, got {e_loss.shape}"
    assert e_items.shape == (3,)
    student.train()


def _run_all():
    fns = [g for name, g in sorted(globals().items()) if name.startswith("test_") and callable(g)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} ultralytics-api tests passed.")


if __name__ == "__main__":
    _run_all()
