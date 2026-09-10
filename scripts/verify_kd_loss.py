"""Measure the KD term against the detection loss on REAL batches.

Written because the claim "KD was too weak to matter" was first argued from
results.csv arithmetic that double-counted Ultralytics' loss gains. v8DetectionLoss
applies box/cls/dfl gains BEFORE returning (utils/loss.py:442-444), so the logged
components are already weighted and must not be multiplied again.

Reports two things the arithmetic cannot settle:
  * the magnitude of each loss component on real post-augmentation batches;
  * the GRADIENT the KD term actually delivers to the student, against the
    detection loss's, over the same trainable parameters. That ratio -- not the
    loss ratio -- is what decides whether distillation steers training.

Usage:
    python scripts/verify_kd_loss.py --lam 1.0 --device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CLASS_NAMES = ["Crazing", "Inclusion", "Patches",
               "Pitted_surface", "Rolled-in_scale", "Scratches"]


def load_batch(img_dir: Path, lbl_dir: Path, n: int, imgsz: int, device: str):
    """One real batch in the dict form v8DetectionLoss expects."""
    import cv2
    imgs = sorted(img_dir.glob("*.jpg"))[:n]
    ims, bidx, cls, boxes = [], [], [], []
    for i, p in enumerate(imgs):
        im = cv2.imread(str(p))
        im = cv2.resize(im, (imgsz, imgsz))
        ims.append(torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0)
        lf = lbl_dir / f"{p.stem}.txt"
        if lf.exists():
            for line in lf.read_text().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                bidx.append(i); cls.append([float(parts[0])])
                boxes.append([float(v) for v in parts[1:5]])
    return {
        "img": torch.stack(ims).to(device),
        "batch_idx": torch.tensor(bidx, dtype=torch.float32, device=device),
        "cls": torch.tensor(cls, dtype=torch.float32, device=device),
        "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", default="experiments/pre_disruption_neu6s_seed0/fl/final_model/client_0_final.pt")
    ap.add_argument("--bank", default="experiments/teacher_bank_neu6s_seed0/teacher_bank")
    ap.add_argument("--kd_weights", default="experiments/competence_across_seeds_neu6s/kd_weights.json")
    ap.add_argument("--data_dir", default="data/neu6s_survivors")
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0])
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--teacher_conf", type=float, nargs="+", default=[0.0],
                    help="Gate the KD term to anchors where the fused teacher "
                         "exceeds this confidence. 0 disables the mask, which "
                         "averages over ~8400 mostly-background anchors.")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils import IterableSimpleNamespace
    from adaptation.kd import KDDetectionLoss
    from adaptation.competence_weights import load_kd_weights
    from model import apply_freeze

    dev = f"cuda:{a.device}" if a.device not in ("", "cpu") else "cpu"

    s = YOLO(a.student)
    student = s.model.to(dev).train()
    student.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    apply_freeze(s, "neck_head")            # same freeze the ablation ran under
    params = [p for p in student.parameters() if p.requires_grad]

    bank = sorted(Path(a.bank).glob("local_c*.pt"), key=lambda q: int(q.stem.split("_c")[1]))
    teachers = []
    for pt in bank:
        t = YOLO(str(pt)).model.to(dev).eval()
        t.requires_grad_(False)
        teachers.append(t)
    names = [p.stem for p in bank]
    lam_c, tw = load_kd_weights(a.kd_weights, CLASS_NAMES, names)
    print(f"student  : {a.student}")
    print(f"teachers : {len(teachers)} from {a.bank}")
    print(f"lambda_c : " + ", ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, lam_c)))

    batch = load_batch(Path(a.data_dir) / "images/train", Path(a.data_dir) / "labels/train",
                       a.batch, a.imgsz, dev)
    print(f"batch    : {batch['img'].shape[0]} real images, {batch['cls'].shape[0]} boxes "
          f"from {a.data_dir}\n")

    hdr = (f"{'lam':>7s}{'t_conf':>8s}{'detect':>10s}"
           f"{'KD':>10s}{'KD share':>10s}{'|g_KD|':>11s}{'|g_det|':>11s}{'grad share':>12s}")
    print(hdr); print("-" * len(hdr))

    combos = [(l, c) for c in a.teacher_conf for l in a.lam]
    for lam, tconf in combos:
        crit = KDDetectionLoss(v8DetectionLoss(student), teachers, lam_c, lam=lam,
                               temperature=a.temperature, student=student,
                               teacher_weights=tw, teacher_conf=tconf)
        student.criterion = crit
        acc = np.zeros(5); gk = gd = 0.0
        for _ in range(a.batches):
            loss_vec, items = student.loss(batch)
            loss_vec = loss_vec / batch["img"].shape[0]     # undo the *batch_size
            box, cls_, dfl = [float(v) for v in items[:3]]
            kd = float(loss_vec[3].detach()) if loss_vec.numel() > 3 else 0.0
            acc += np.array([box, cls_, dfl, box + cls_ + dfl, kd])
            det = loss_vec[:3].sum()
            g_det = torch.autograd.grad(det, params, retain_graph=True, allow_unused=True)
            gd += float(torch.sqrt(sum((g.pow(2).sum() for g in g_det if g is not None))))
            if loss_vec.numel() > 3:
                g_kd = torch.autograd.grad(loss_vec[3], params, retain_graph=False, allow_unused=True)
                gk += float(torch.sqrt(sum((g.pow(2).sum() for g in g_kd if g is not None))))
        acc /= a.batches; gk /= a.batches; gd /= a.batches
        share = acc[4] / (acc[3] + acc[4]) * 100 if (acc[3] + acc[4]) else 0.0
        gshare = gk / (gk + gd) * 100 if (gk + gd) else 0.0
        print(f"{lam:>7.1f}{tconf:>8.2f}{acc[3]:>10.3f}"
              f"{acc[4]:>10.4f}{share:>9.2f}%{gk:>11.4f}{gd:>11.4f}{gshare:>11.2f}%")

    print("\ngrad share is the number that matters: the fraction of the update's")
    print("magnitude that distillation contributes, over the trainable (neck+head) params.")


if __name__ == "__main__":
    main()
