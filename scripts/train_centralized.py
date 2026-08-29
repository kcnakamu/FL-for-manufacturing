"""
Centralized YOLOv8n training.

Modes
-----
head_only : backbone + neck frozen, only detection head trains
neck_head : backbone frozen, neck + head train
full      : entire model trains

Usage
-----
# Centralized baseline (from scratch, all data pooled)
python scripts/train_centralized.py \
    --data data/neu_data/all_clients/data.yaml --mode full --epochs 150 --lr 0.01 \
    --output_dir experiments/disruption_neu_fedavg/baselines/centralized

# Stage 1 — head-only adaptation from the Round-10 global model
python scripts/train_centralized.py \
    --data data/neu_data/client_2/data.yaml \
    --weights experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
    --mode head_only --epochs 25 --lr 0.001 \
    --output_dir experiments/disruption_neu_fedavg/adaptation

# Stage 2 — neck+head fine-tuning from the Stage-1 checkpoint
python scripts/train_centralized.py \
    --data data/neu_data/client_2/data.yaml \
    --weights experiments/disruption_neu_fedavg/adaptation/head_only/weights/best.pt \
    --mode neck_head --epochs 75 --lr 0.0001 \
    --output_dir experiments/disruption_neu_fedavg/adaptation
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Reuse the shared freeze helper so backbone/neck/head boundaries live in one place.
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from model import (  # noqa: E402
    apply_freeze as _apply_freeze,
    freeze_indices as _freeze_indices,
    load_model as _load_model,
)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible head init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(weights: str | None, num_classes: int) -> YOLO:
    """Load a checkpoint and adapt the head to num_classes, KEEPING pretrained weights.

    This used to replace model.model with a fresh DetectionModel whenever the
    class count differed, which silently discarded every pretrained tensor --
    backbone and neck included. Starting from yolov8n.pt (nc=80) with any other
    num_classes therefore trained from random init while looking like transfer
    learning, and initialization dominates results on this dataset.

    model.load_model() does the transfer properly: every shape-matching tensor is
    copied and only the head is randomly initialized. Note this changes behaviour
    for runs that hit the nc-mismatch path before this fix.
    """
    if weights is None:
        return _load_model(num_classes=num_classes)

    model = YOLO(weights)
    if model.model.nc == num_classes:
        return model

    pretrained = model.model.state_dict()
    new_model = DetectionModel(model.model.yaml, nc=num_classes).to(model.device)
    target = new_model.state_dict()
    transfer = {k: v for k, v in pretrained.items()
                if k in target and v.shape == target[k].shape}
    new_model.load_state_dict(transfer, strict=False)
    print(f"[_build_model] Adapted {weights} head to nc={num_classes}: "
          f"transferred {len(transfer)} tensors, "
          f"{len(target) - len(transfer)} randomly initialized (head).")
    new_model.nc = num_classes
    model.model = new_model
    return model


def train(
    data: str,
    mode: str,
    epochs: int,
    lr: float,
    weights: str | None,
    num_classes: int,
    imgsz: int,
    batch: int,
    output_dir: str,
    device: str,
    workers: int,
    seed: int,
) -> None:
    # Seed before building the model so the random detection-head init
    # (when adapting yolov8n's nc=80 head to num_classes) is reproducible.
    set_seed(seed)
    model = _build_model(weights, num_classes)
    _apply_freeze(model, mode)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        optimizer="SGD",
        workers=workers,
        device=device,
        project=str(out_path.resolve()),
        name=mode,
        # The trainer re-enables requires_grad for params not listed here, so
        # the freeze must go through this arg — apply_freeze alone is undone.
        freeze=_freeze_indices(mode),
        seed=seed,
    )

    summary = {
        "mode": mode,
        "epochs": epochs,
        "lr": lr,
        "weights": weights,
        "num_classes": num_classes,
        "imgsz": imgsz,
        "batch": batch,
        "data": data,
        "seed": seed,
    }
    summary_path = Path(model.trainer.save_dir) / "train_config.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Config saved to {summary_path}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Centralized YOLOv8n training.")
    parser.add_argument("--data", required=True, help="Path to dataset YAML")
    parser.add_argument("--mode", choices=["head_only", "neck_head", "full"], default="full")
    parser.add_argument("--weights", default=None, help="Starting .pt checkpoint (omit to use yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output_dir", default="adaptation",
                        help="Output dir for training runs. Pass experiments/<exp_name>/adaptation (or baselines/<name>).")
    parser.add_argument("--device", default="", help="Device (e.g. '0', 'cpu'). Empty = auto.")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for head init + YOLO training. Vary across runs "
                             "to test robustness; match the FL run's seed for a full-experiment repeat.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data=args.data,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        weights=args.weights,
        num_classes=args.num_classes,
        imgsz=args.imgsz,
        batch=args.batch,
        output_dir=args.output_dir,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
    )
