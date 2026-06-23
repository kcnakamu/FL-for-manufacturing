"""
Centralized YOLOv8n training.

Modes
-----
head_only : backbone + neck frozen, only detection head trains
neck_head : backbone frozen, neck + head train
full      : entire model trains

Usage
-----
# From scratch
python "centralized_training/train_centralized.py" \
    --data dataset/data.yaml --mode full --epochs 150 --lr 0.01

# From checkpoint
python "centralized_training/train_centralized.py" \
    --data dataset/data.yaml --weights runs/exp/weights/best.pt \
    --mode neck_head --epochs 30 --lr 0.001

# my use
python "centralized_training/train_centralized.py"  --data datasets/neu_data/client_2/data.yaml  --weights fl_runs/20260606_170422_fedavg_disruption/final_model/client_0_final.pt --mode head_only --epochs 25 --lr 0.001 --output_dir centralized_training/runs

python "centralized_training/train_centralized.py"  --data datasets/neu_data/client_2/data.yaml  --weights /home/kcnakamu/s26_urop/FL-for-manufacturing/centralized_training/runs/head_only/weights/best.pt --mode neck_head --epochs 75 --lr 0.0001 --output_dir centralized_training/runs

python "centralized_training/train_centralized.py" --data neu_data/client_0/data.yaml --mode full --epochs 150 --lr 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# YOLOv8n layer indices: backbone 0-9, neck 10-21, head 22
# https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers#freeze-all-except-final-detection-layers
_BACKBONE_END = 10
_NECK_END = 22


def _build_model(weights: str | None, num_classes: int) -> YOLO:
    model = YOLO(weights if weights is not None else "yolov8n.pt")
    # Rebuild detection head if the checkpoint's class count doesn't match (e.g. yolov8n.pt is nc=80)
    if model.model.nc != num_classes:
        model.model = DetectionModel(model.model.yaml, nc=num_classes).to(model.device)
    return model


def _apply_freeze(model: YOLO, mode: str) -> None:
    layers = list(model.model.model)
    if mode == "head_only":
        freeze_up_to = _NECK_END
    elif mode == "neck_head":
        freeze_up_to = _BACKBONE_END
    else:
        freeze_up_to = 0

    for i, layer in enumerate(layers):
        requires_grad = i >= freeze_up_to
        for param in layer.parameters():
            param.requires_grad = requires_grad

    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"Mode '{mode}': {trainable:,} / {total:,} trainable ({100 * trainable / total:.1f}%)")


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
) -> None:
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
        freeze=[], 
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
    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--output_dir", default="adaptation",
                        help="Output dir for training runs. Pass experiments/<exp_name>/adaptation (or baselines/<name>).")
    parser.add_argument("--device", default="", help="Device (e.g. '0', 'cpu'). Empty = auto.")
    parser.add_argument("--workers", type=int, default=0)
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
    )
