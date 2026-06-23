"""
Evaluate model checkpoints across fine-tuning stages.

Stages
------
round10 : Round 10 global FL model (before disruption)
stage1  : After stage-1 head-only fine-tuning
stage2  : After stage-2 neck+head fine-tuning

Datasets
--------
client2_data : client 2 validation YAML  (uses the 'val' split)
central_data : central test YAML          (uses the 'test' split)

Metrics
-------
mAP50, Precision, Recall, F1, False Positives, FP per image

Usage
-----
python "centralized training/analyze_results.py" \
    --round10      path/to/round10.pt \
    --stage1       path/to/stage1_best.pt \
    --stage2       path/to/stage2_best.pt \
    --client2_data dataset/client2/dataset.yaml \
    --central_data dataset/central_test/dataset.yaml \
    --output       runs/analysis/results.json

python "centralized_training/analyze_results.py" \
    --round10     fl_runs/20260606_170422_fedavg_disruption/final_model/client_0_final.pt \
    --stage1      centralized_training/runs/head_only/weights/best.pt \
    --stage2      centralized_training/runs/neck_head/weights/best.pt \
    --client2_data  datasets/neu_data/client_2/data.yaml \
    --central_data  datasets/neu_data/test/data.yaml \
    --output       centralized_training/analysis/results.json

"""


from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _count_images(data_yaml: str, split: str) -> int:
    with open(data_yaml) as f:
        d = yaml.safe_load(f)
    root = Path(d.get("path", Path(data_yaml).parent))
    split_path = root / d[split]
    return len([f for f in split_path.rglob("*") if f.suffix.lower() in _IMAGE_SUFFIXES])


def run_val(model_path: str, data: str, split: str, device: str) -> dict:
    model = YOLO(model_path)
    # plots=True is required for Ultralytics ≥8.x to populate the confusion matrix;
    # process_batch() is gated behind `if self.args.plots` in DetectionValidator.
    results = model.val(data=data, split=split, device=device, verbose=False, plots=True)

    mp = float(results.box.mp)
    mr = float(results.box.mr)
    map50 = float(results.box.map50)
    f1 = 2 * mp * mr / (mp + mr + 1e-8)

    # Confusion matrix: rows = predicted class, cols = true class.
    # Last index (nc) is the background/"no match" sentinel.
    #   matrix[i, i]       → TP for class i
    #   matrix[i, :].sum() - matrix[i, i] → FP for class i (predicted i, wrong GT)
    #   matrix[:, i].sum() - matrix[i, i] → FN for class i (GT is i, not predicted as i)
    cm = results.confusion_matrix.matrix  # shape (nc+1, nc+1)
    nc = cm.shape[0] - 1
    names = results.names

    total_fp = int(cm[:-1, -1].sum())

    per_class_confusion: dict = {}
    for i in range(nc):
        tp = int(cm[i, i])
        fp = int(cm[i, :].sum() - cm[i, i])
        fn = int(cm[:, i].sum() - cm[i, i])
        per_class_confusion[names[i]] = {"tp": tp, "fp": fp, "fn": fn}

    n_images = _count_images(data, split)
    fp_per_image = total_fp / n_images if n_images > 0 else float("nan")

    return {
        "map50": round(map50, 4),
        "precision": round(mp, 4),
        "recall": round(mr, 4),
        "f1": round(f1, 4),
        "false_positives": total_fp,
        "fp_per_image": round(fp_per_image, 4),
        "n_images": n_images,
        "per_class_confusion": per_class_confusion,
    }


def analyze(
    round10: str,
    stage1: str,
    stage2: str,
    client2_data: str,
    central_data: str,
    device: str,
    output: str,
) -> None:
    stages = {
        "round10_global":    (round10, "Round 10 global FL model"),
        "stage1_head_only":  (stage1,  "Stage 1: head-only fine-tuning"),
        "stage2_neck_head":  (stage2,  "Stage 2: neck+head fine-tuning"),
    }
    datasets = {
        "client2_val":  (client2_data, "val"),
        "central_test": (central_data, "val"),
    }

    all_results: dict = {}
    for stage_key, (model_path, stage_label) in stages.items():
        all_results[stage_key] = {}
        for dataset_key, (data_yaml, split) in datasets.items():
            print(f"\n[{stage_label}] — {dataset_key} ...")
            metrics = run_val(model_path, data_yaml, split, device)
            all_results[stage_key][dataset_key] = metrics
            print(
                f"  mAP50={metrics['map50']:.4f}  "
                f"P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  "
                f"F1={metrics['f1']:.4f}  "
                f"FP={metrics['false_positives']}  "
                f"FP/img={metrics['fp_per_image']:.4f}"
            )
            print(f"  Confusion  {'Class':<20} {'TP':>6} {'FP':>6} {'FN':>6}")
            for cls_name, counts in metrics["per_class_confusion"].items():
                print(f"    {cls_name:<20} {counts['tp']:>6} {counts['fp']:>6} {counts['fn']:>6}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuning stages.")
    parser.add_argument("--round10",      required=True, help="Round 10 global FL model (.pt)")
    parser.add_argument("--stage1",       required=True, help="Stage 1 best checkpoint (.pt)")
    parser.add_argument("--stage2",       required=True, help="Stage 2 best checkpoint (.pt)")
    parser.add_argument("--client2_data", required=True, help="Client 2 dataset YAML")
    parser.add_argument("--central_data", required=True, help="Central test dataset YAML (must have 'test' split)")
    parser.add_argument("--device",       default="",    help="Device (e.g. '0', 'cpu'). Empty = auto.")
    parser.add_argument("--output",       default="analysis/results.json",
                        help="Output JSON path. Pass experiments/<exp_name>/analysis/results.json.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze(
        round10=args.round10,
        stage1=args.stage1,
        stage2=args.stage2,
        client2_data=args.client2_data,
        central_data=args.central_data,
        device=args.device,
        output=args.output,
    )
