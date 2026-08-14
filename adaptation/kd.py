"""Class-retention-weighted output-level knowledge distillation for YOLOv8.

Manuscript contribution 3: retain knowledge learned from the unavailable
clients (A, B) without accessing their raw images. While C fine-tunes on its
own data, a frozen teacher (the pre-disruption global model) runs on the same
batches and the student's per-anchor class logits are pulled toward the
teacher's -- weighted per class by how much contribution the offline clients
stand to lose (class_retention_weights.json from shapley/contribution.py):

    total = box + cls + dfl + lambda * sum_c w_c * KD_c

Integration (verified against ultralytics 8.4.33):
  * DetectionModel.loss() uses a pre-set `self.criterion` -> we inject
    KDDetectionLoss without touching any forward code.
  * Teacher and student share the architecture and imgsz, so their per-anchor
    class-logit tensors (b, nc, A) are anchor-aligned -- dense response
    distillation with no box matching / NMS.
  * The EMA (what gets validated/saved) is deep-copied in _setup_train BEFORE
    the criterion is attached, so the teacher is never serialized into
    checkpoints.

The KD math (`kd_class_term`) is pure and CPU-tested in
adaptation/tests/test_kd.py; the pinned Ultralytics API surface is tripwired in
adaptation/tests/test_ultralytics_api.py.

Fallback if this API shifts in a future Ultralytics: response distillation via
teacher pseudo-labels (predict on C's images, add high-confidence detections as
extra label lines with per-class thresholds from w_c) needs no loss surgery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


def kd_class_term(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  class_weights: torch.Tensor, temperature: float = 2.0,
                  teacher_conf: float = 0.0):
    """Per-class response-distillation term on anchor-aligned class logits.

    Args:
        student_logits: (b, nc, A) raw class logits from the student head.
        teacher_logits: (b, nc, A) raw class logits from the frozen teacher.
        class_weights:  (nc,) nonnegative weights (normalized upstream).
        temperature:    softening T; loss scaled by T^2 as usual.
        teacher_conf:   if > 0, only anchors where the teacher's max class
                        probability exceeds this contribute (suppresses the
                        background-anchor flood).

    Returns:
        (kd, kd_per_class): scalar sum_c w_c * KD_c, and the detached (nc,)
        per-class means for logging.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(f"student {tuple(student_logits.shape)} vs teacher "
                         f"{tuple(teacher_logits.shape)} logits shape mismatch.")
    T = float(temperature)
    soft_targets = torch.sigmoid(teacher_logits.detach() / T)
    per_elem = F.binary_cross_entropy_with_logits(
        student_logits / T, soft_targets, reduction="none") * (T * T)  # (b, nc, A)

    if teacher_conf > 0.0:
        mask = (torch.sigmoid(teacher_logits.detach()).amax(dim=1, keepdim=True)
                > teacher_conf).to(per_elem.dtype)          # (b, 1, A)
        denom = mask.sum().clamp(min=1.0)
        kd_per_class = (per_elem * mask).sum(dim=(0, 2)) / denom
    else:
        kd_per_class = per_elem.mean(dim=(0, 2))            # (nc,)

    w = class_weights.to(kd_per_class.dtype).to(kd_per_class.device)
    kd = (w * kd_per_class).sum()
    return kd, kd_per_class.detach()


class KDDetectionLoss:
    """Wraps v8DetectionLoss, appending a weighted KD term as a 4th loss item.

    Mirrors the base criterion's contract exactly: returns
    (loss_vector * batch_size, detached_loss_vector), now length 4
    (box, cls, dfl, kd).
    """

    def __init__(self, base, teacher: torch.nn.Module, class_weights,
                 lam: float = 1.0, temperature: float = 2.0,
                 teacher_conf: float = 0.0):
        self.base = base
        self.teacher = teacher
        self.w = torch.as_tensor(list(class_weights), dtype=torch.float32)
        self.lam = float(lam)
        self.temperature = float(temperature)
        self.teacher_conf = float(teacher_conf)

    def _teacher_scores(self, img: torch.Tensor) -> torch.Tensor:
        device_type = img.device.type
        # Teacher stays fp32 outside autocast for numerical stability under AMP.
        with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
            out = self.teacher(img.float())
        preds = out[1] if isinstance(out, tuple) else out  # eval mode -> (y, preds)
        return preds["scores"]

    def __call__(self, preds, batch):
        preds = self.base.parse_output(preds)
        _, loss3, loss3_detached = self.base.get_assigned_targets_and_loss(preds, batch)
        batch_size = preds["boxes"].shape[0]

        # KD is a training-only term. During validation/inference Ultralytics
        # runs under no_grad and accumulates loss into a 3-wide tensor (box, cls,
        # dfl), so returning a 4th item there crashes the validator. Gate on
        # grad-enabled: training -> 4 items, validation -> plain 3-item loss.
        if not torch.is_grad_enabled():
            return loss3 * batch_size, loss3_detached

        t_scores = self._teacher_scores(batch["img"])
        kd, _ = kd_class_term(preds["scores"], t_scores.to(preds["scores"].dtype),
                              self.w, self.temperature, self.teacher_conf)
        kd = self.lam * kd

        loss = torch.cat([loss3, kd.view(1)])
        loss_detached = torch.cat([loss3_detached, kd.detach().view(1)])
        return loss * batch_size, loss_detached


def load_class_weights(path, class_names: List[str]) -> List[float]:
    """Read class_retention_weights.json (shapley/contribution.py) -> ordered list.

    Validates that the file covers exactly the given classes and renormalizes.
    """
    with open(path) as fh:
        data = json.load(fh)
    weights: Dict[str, float] = data["weights"] if "weights" in data else data
    missing = [c for c in class_names if c not in weights]
    if missing:
        raise ValueError(f"{path} is missing weights for classes {missing} "
                         f"(has {sorted(weights)}).")
    vals = [float(weights[c]) for c in class_names]
    if any(v < 0 for v in vals):
        raise ValueError(f"Negative class weights in {path}: {weights}")
    s = sum(vals)
    if s <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [v / s for v in vals]


def make_kd_trainer(teacher_pt: str, class_weights: List[float], lam: float = 1.0,
                    temperature: float = 2.0, teacher_conf: float = 0.0):
    """Build a DetectionTrainer subclass that attaches the KD criterion.

    Passed to YOLO(...).train(trainer=<this>, ...). The criterion is injected
    after super()._setup_train() so the EMA snapshot (validated + saved) is
    copied before the teacher is attached.
    """
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import unwrap_model

    class KDDetectionTrainer(DetectionTrainer):
        def _setup_train(self):
            super()._setup_train()
            teacher = YOLO(teacher_pt).model.to(self.device).eval()
            teacher.requires_grad_(False)
            student = unwrap_model(self.model)
            student.criterion = KDDetectionLoss(
                v8DetectionLoss(student), teacher, class_weights,
                lam=lam, temperature=temperature, teacher_conf=teacher_conf)
            # get_validator() (called inside super()) resets this to 3 names.
            self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "kd_loss")

    return KDDetectionTrainer
