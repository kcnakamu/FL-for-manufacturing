"""
Evaluate a trained YOLO model on the global test set at one or more NMS
(IoU) thresholds.

Ultralytics applies NMS with a default IoU threshold of 0.7. This script lets
you evaluate at a chosen threshold (e.g. 0.4) or sweep several values to see
how the NMS overlap threshold affects P/R/F1/mAP.

Usage:
    # single NMS threshold
    python utils/analysis/evaluate_nms.py \
        --model experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
        --data_dir data/neu_data --iou 0.4

    # sweep several NMS thresholds (prints a comparison table)
    python utils/analysis/evaluate_nms.py --model best.pt \
        --data_dir data/neu_data --iou 0.3 0.4 0.5 0.6 0.7

    # save the sweep to CSV
    python utils/analysis/evaluate_nms.py --model best.pt \
        --data_dir data/neu_data --iou 0.4 0.5 0.6 --save_csv nms_sweep.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Reuse the test-set evaluation machinery so behaviour stays identical to
# evaluate_test.py (same imgsz/batch, same metric extraction).
from evaluate_test import (
    RESULT_HEADER,
    build_test_yaml,
    eval_model,
    format_result_row,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLO model on the test set across NMS (IoU) thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Path to a .pt weight file")
    parser.add_argument("--iou", nargs="+", type=float, required=True,
                        help="One or more NMS IoU thresholds to evaluate, e.g. 0.4 0.5 0.6")
    parser.add_argument("--data_dir",    default="data",
                        help="Root data directory containing the 'test/' subfolder")
    parser.add_argument("--class_names", nargs="+",
                        default=["Inclusion", "Patches", "Scratches"],
                        help="YOLO class names (order must match label class IDs)")
    parser.add_argument("--output_dir",  default="test_results_nms",
                        help="Directory for YOLO val output")
    parser.add_argument("--save_csv",    default=None,
                        help="Optional path to save results as CSV")
    parser.add_argument("--device",      default=None,
                        help="Device override, e.g. 'cpu', '0' (auto-detected if omitted)")
    parser.add_argument("--conf",        type=float, default=None,
                        help="Confidence threshold for predictions "
                             "(default: Ultralytics val default of 0.001)")
    parser.add_argument("--per_class",   action="store_true",
                        help="Also report per-class mAP50 and mAP50-95")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.exists():
        sys.exit(f"Test directory not found: {test_dir}")

    test_yaml = build_test_yaml(test_dir, args.class_names)
    model_tag = Path(args.model).parent.parent.name
    print(f"Model:    {args.model}")
    print(f"Test set: {test_dir.resolve()}")
    print(f"Device:   {device}")
    print(f"NMS IoU thresholds: {args.iou}\n")

    rows = []
    labels = []
    for i, iou in enumerate(args.iou):
        # Separate output subdir per threshold so YOLO plots don't overwrite.
        out_dir = Path(args.output_dir) / f"{model_tag}_iou{iou}"
        label = f"iou={iou:.2f}"
        print(f"[{i+1}/{len(args.iou)}] Evaluating with NMS {label}")
        row = eval_model(args.model, test_yaml, out_dir, device,
                         conf=args.conf, iou=iou, per_class=args.per_class)
        row["iou"] = iou
        rows.append(row)
        labels.append(label)
        print(
            f"  P={row['P']:.4f}  R={row['R']:.4f}  F1={row['F1']:.4f}"
            f"  mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50-95']:.4f}"
        )
        if args.per_class:
            print("\n" + RESULT_HEADER)
            print("-" * len(RESULT_HEADER))
            for pc in row["per_class"]:
                print(format_result_row(pc["class"], pc))
            print(format_result_row("all", row))
            print()

    Path(test_yaml).unlink(missing_ok=True)

    # Comparison table across thresholds.
    print("\n" + RESULT_HEADER)
    print("-" * len(RESULT_HEADER))
    for label, row in zip(labels, rows):
        print(format_result_row(label, row))
    print()

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["iou", "model", "P", "R", "F1", "mAP50", "mAP50-95"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
