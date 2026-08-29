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
    t = teacher_logits.detach()
    return kd_class_term_from_target(
        student_logits, torch.sigmoid(t / float(temperature)), class_weights,
        temperature=temperature,
        mask_probs=torch.sigmoid(t) if teacher_conf > 0.0 else None,
        teacher_conf=teacher_conf,
    )


def fuse_teacher_probs(teacher_logits, teacher_weights: torch.Tensor,
                       temperature: float = 2.0) -> torch.Tensor:
    """Weighted-mean of per-teacher class probabilities -- the ensemble target.

        p_ens[i,c,a] = sum_k w[k,c] * sigmoid(z_k[i,c,a] / T)

    Because YOLOv8's class head is per-class sigmoid (multi-label), each term is
    a Bernoulli parameter and a convex combination of them is one too -- so the
    result needs no renormalisation. Averaging PROBABILITIES, not logits: sigmoid
    is nonlinear, so averaging logits would not give the mixture probability and
    would have no probabilistic reading.

    Teachers share architecture and imgsz, so anchor index `a` refers to the same
    location and scale in every one of them; that is what lets these be combined
    elementwise with no matching or NMS.

    Args:
        teacher_logits: sequence of K tensors, each (b, nc, A).
        teacher_weights: (K, nc), each class column summing to 1.
        temperature: pass 1.0 to get unsoftened probabilities (for conf masking).
    """
    zs = list(teacher_logits)
    if not zs:
        raise ValueError("fuse_teacher_probs needs at least one teacher")

    K, ref = len(zs), zs[0]
    for k, z in enumerate(zs):
        if z.shape != ref.shape:
            raise ValueError(f"teacher {k} logits {tuple(z.shape)} != teacher 0 "
                             f"{tuple(ref.shape)}; teachers must share arch and imgsz")
    nc = ref.shape[1]
    if tuple(teacher_weights.shape) != (K, nc):
        raise ValueError(f"teacher_weights {tuple(teacher_weights.shape)} != "
                         f"(K={K}, nc={nc})")

    col_sums = teacher_weights.sum(dim=0)
    if not torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-4):
        raise ValueError(
            f"teacher_weights columns must sum to 1 (got {col_sums.tolist()}). "
            "Unnormalised weights break the convex-combination property that makes "
            "the fused target a valid probability."
        )

    T = float(temperature)
    out = None
    for k, z in enumerate(zs):
        wk = teacher_weights[k].to(device=z.device, dtype=z.dtype).view(1, nc, 1)
        term = wk * torch.sigmoid(z.detach() / T)
        out = term if out is None else out + term
    return out


def kd_class_term_from_target(student_logits: torch.Tensor, target_probs: torch.Tensor,
                              class_weights: torch.Tensor, temperature: float = 2.0,
                              mask_probs: Optional[torch.Tensor] = None,
                              teacher_conf: float = 0.0):
    """Per-class distillation against an arbitrary soft target.

    Shared by the single-teacher path (target = sigmoid(z/T)) and the ensemble
    path (target = fuse_teacher_probs(...)).

    Args:
        student_logits: (b, nc, A) raw class logits from the student head.
        target_probs:   (b, nc, A) soft targets in [0, 1], already temperature-scaled.
        class_weights:  (nc,) lambda_c -- how much each class contributes.
        mask_probs:     (b, nc, A) UNSOFTENED probabilities used for the
                        confidence mask; defaults to target_probs. Kept separate
                        because the mask should reflect real confidence, not the
                        temperature-flattened target (at high T every anchor
                        drifts toward 0.5 and the mask stops discriminating).
        teacher_conf:   if > 0, only anchors whose max probability exceeds this
                        contribute -- suppresses the background-anchor flood.
    """
    if student_logits.shape != target_probs.shape:
        raise ValueError(f"student {tuple(student_logits.shape)} vs target "
                         f"{tuple(target_probs.shape)} shape mismatch.")
    T = float(temperature)
    per_elem = F.binary_cross_entropy_with_logits(
        student_logits / T, target_probs.detach(), reduction="none") * (T * T)

    if teacher_conf > 0.0:
        src = target_probs if mask_probs is None else mask_probs
        mask = (src.detach().amax(dim=1, keepdim=True)
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

    def __init__(self, base, teacher, class_weights,
                 lam: float = 1.0, temperature: float = 2.0,
                 teacher_conf: float = 0.0, student: Optional[torch.nn.Module] = None,
                 teacher_weights=None):
        """
        Args:
            teacher: one nn.Module, or a sequence of them (the teacher bank).
            class_weights: (nc,) lambda_c -- how much to distill each class.
            teacher_weights: (K, nc) w[k,c] -- which teacher to believe per class,
                columns summing to 1. None means uniform 1/K, i.e. unweighted
                ensembling; with a single teacher that is the same thing as the
                original single-teacher behaviour.
        """
        self.base = base
        self.teachers = ([teacher] if isinstance(teacher, torch.nn.Module)
                         else list(teacher))
        if not self.teachers:
            raise ValueError("at least one teacher is required")
        self.student = student  # to detect eval mode during validation
        self.w = torch.as_tensor(list(class_weights), dtype=torch.float32)
        nc, K = self.w.numel(), len(self.teachers)

        if teacher_weights is None:
            self.tw = torch.full((K, nc), 1.0 / K, dtype=torch.float32)
        else:
            self.tw = torch.as_tensor(teacher_weights, dtype=torch.float32)
            if tuple(self.tw.shape) != (K, nc):
                raise ValueError(f"teacher_weights {tuple(self.tw.shape)} != "
                                 f"(K={K}, nc={nc})")

        self.lam = float(lam)
        self.temperature = float(temperature)
        self.teacher_conf = float(teacher_conf)
        self.last_kd = float("nan")  # last training-batch KD term, for logging

    @property
    def teacher(self):
        """Back-compat for callers written against the single-teacher class."""
        return self.teachers[0]

    def _teacher_scores(self, img: torch.Tensor) -> list:
        """Class logits from every teacher on the same (augmented) batch.

        Must run online: batch["img"] is post-augmentation, so the teachers have
        to see exactly the tensor the student saw. Nothing can be precomputed.
        """
        device_type = img.device.type
        out_list = []
        for t in self.teachers:
            # Teachers stay fp32 outside autocast for numerical stability under AMP.
            with torch.no_grad(), torch.autocast(device_type=device_type, enabled=False):
                out = t(img.float())
            preds = out[1] if isinstance(out, tuple) else out  # eval -> (y, preds)
            out_list.append(preds["scores"])
        return out_list

    def __call__(self, preds, batch):
        preds = self.base.parse_output(preds)
        _, loss3, loss3_detached = self.base.get_assigned_targets_and_loss(preds, batch)
        batch_size = preds["boxes"].shape[0]

        # The tracked loss vector (2nd return) stays 3-wide (box, cls, dfl) in
        # EVERY path. Ultralytics sizes the validator's loss accumulator from
        # trainer.loss_items and validates the KD-free EMA model, so a 4-wide
        # tracked vector would crash validation with a 4-vs-3 mismatch. KD is
        # instead folded ONLY into the summed loss (1st return) that the trainer
        # backpropagates via loss.sum() -- so it still trains the student.
        training_forward = torch.is_grad_enabled() and (
            self.student is None or self.student.training)
        if not training_forward:
            return loss3 * batch_size, loss3_detached

        s_scores = preds["scores"]
        t_scores = [z.to(s_scores.dtype) for z in self._teacher_scores(batch["img"])]
        tw = self.tw.to(s_scores.device)

        target = fuse_teacher_probs(t_scores, tw, self.temperature)
        # Mask on UNSOFTENED ensemble confidence: at T>1 every anchor drifts
        # toward 0.5 and a mask taken on the softened target stops discriminating.
        mask_probs = (fuse_teacher_probs(t_scores, tw, 1.0)
                      if self.teacher_conf > 0.0 else None)
        kd, _ = kd_class_term_from_target(
            s_scores, target, self.w, self.temperature,
            mask_probs=mask_probs, teacher_conf=self.teacher_conf)
        kd = self.lam * kd
        self.last_kd = float(kd.detach())

        loss = torch.cat([loss3, kd.view(1)])          # 4-wide: summed for backprop
        return loss * batch_size, loss3_detached       # tracked vector stays 3-wide


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


def make_kd_trainer(teacher_pt, class_weights: List[float], lam: float = 1.0,
                    temperature: float = 2.0, teacher_conf: float = 0.0,
                    teacher_weights=None):
    """Build a DetectionTrainer subclass that attaches the KD criterion.

    Passed to YOLO(...).train(trainer=<this>, ...). The criterion is injected
    after super()._setup_train() so the EMA snapshot (validated + saved) is
    copied before the teacher is attached.
    """
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import LOGGER
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.torch_utils import unwrap_model

    def _log_kd(trainer):
        kd = getattr(unwrap_model(trainer.model), "criterion", None)
        if kd is not None and hasattr(kd, "last_kd"):
            LOGGER.info(f"KD term (last train batch): {kd.last_kd:.4f}")

    class KDDetectionTrainer(DetectionTrainer):
        def _setup_train(self):
            super()._setup_train()
            paths = ([teacher_pt] if isinstance(teacher_pt, (str, Path))
                     else list(teacher_pt))
            student = unwrap_model(self.model)
            teachers = []
            for pt in paths:
                t = YOLO(str(pt)).model.to(self.device).eval()
                t.requires_grad_(False)
                if t.nc != student.nc:
                    raise ValueError(
                        f"teacher {pt} has nc={t.nc} but the student has "
                        f"nc={student.nc}; class logits must be element-wise "
                        f"comparable for dense distillation."
                    )
                teachers.append(t)
            LOGGER.info(f"KD: {len(teachers)} teacher(s) attached "
                        f"({'weighted' if teacher_weights is not None else 'uniform'})")
            student.criterion = KDDetectionLoss(
                v8DetectionLoss(student), teachers, class_weights,
                lam=lam, temperature=temperature, teacher_conf=teacher_conf,
                student=student, teacher_weights=teacher_weights)
            # NOTE: loss_names stays 3 (box, cls, dfl). KD is folded into the
            # backpropped total, not tracked as a 4th item -- see KDDetectionLoss
            # for why the tracked vector must stay 3-wide. The KD term is logged
            # per epoch via the callback below instead of a progress column.
            self.add_callback("on_fit_epoch_end", _log_kd)

    return KDDetectionTrainer
