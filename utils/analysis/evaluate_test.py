"""
Evaluate a trained YOLO model on the global test set.

Usage:
    # single model
    python utils/analysis/evaluate_test.py \
        --model experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
        --data_dir data/neu_data --class_names Inclusion Patches Scratches

    # multiple models (prints a comparison table)
    python utils/analysis/evaluate_test.py --model a/weights/best.pt b/weights/best.pt --names A B

    # save results to CSV
    python utils/analysis/evaluate_test.py --model best.pt --data_dir data/neu_data --save_csv results.csv
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import yaml


def build_test_yaml(test_dir: Path, class_names: list[str]) -> str:
    """Write a temporary data.yaml pointing at the test set and return its path."""
    data = {
        "path":  str(test_dir.resolve()),
        "train": "images", 
        "val":   "images",  
        "test":  "images",
        "nc":    len(class_names),
        "names": class_names,
    }
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="test_eval_"
    )
    yaml.dump(data, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    return tmp.name


def eval_model(model_path: str, test_yaml: str, output_dir: Path, device: str,
               conf: float | None = None, iou: float | None = None,
               per_class: bool = False) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ultralytics uses iou=0.7 for NMS by default; only override when supplied.
    val_kwargs = {}
    if iou is not None:
        val_kwargs["iou"] = iou

    metrics = model.val(
        data=test_yaml,
        split="test",
        imgsz=480,
        batch=16,
        workers=0,
        verbose=per_class,
        device=device,
        conf=conf,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        **val_kwargs,
    )

    p  = float(metrics.box.mp)
    r  = float(metrics.box.mr)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    row = {
        "model":    model_path,
        "P":        p,
        "R":        r,
        "F1":       f1,
        "mAP50":    float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }

    if per_class:
        # ap_class_index lists only the classes present in the test set, so
        # index the per-class arrays through it rather than assuming 0..nc-1.
        names = model.names
        per_class_rows = []
        for i, c in enumerate(metrics.box.ap_class_index):
            pc_p = float(metrics.box.p[i])
            pc_r = float(metrics.box.r[i])
            pc_f1 = (2 * pc_p * pc_r / (pc_p + pc_r)) if (pc_p + pc_r) > 0 else 0.0
            per_class_rows.append({
                "class":    names[int(c)],
                "P":        pc_p,
                "R":        pc_r,
                "F1":       pc_f1,
                "mAP50":    float(metrics.box.ap50[i]),
                "mAP50-95": float(metrics.box.ap[i]),
            })
        row["per_class"] = per_class_rows

    return row


# Shared column layout so the summary table and the per-class blocks align.
RESULT_HEADER = (
    f"{'Name':<30}  {'P':>7}  {'R':>7}  {'F1':>7}  {'mAP50':>7}  {'mAP50-95':>9}"
)


def format_result_row(label: str, row: dict) -> str:
    return (
        f"{label:<30}  {row['P']:>7.4f}  {row['R']:>7.4f}  {row['F1']:>7.4f}"
        f"  {row['mAP50']:>7.4f}  {row['mAP50-95']:>9.4f}"
    )


def print_table(rows: list[dict], names: list[str]):
    print("\n" + RESULT_HEADER)
    print("-" * len(RESULT_HEADER))
    for name, row in zip(names, rows):
        print(format_result_row(name, row))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO model(s) on the global test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", nargs="+", required=True,
        help="Path(s) to .pt weight file(s)",
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Display name for each model (defaults to the model path)",
    )
    parser.add_argument("--data_dir",    default="data",
                        help="Root data directory containing the 'test/' subfolder")
    parser.add_argument("--class_names", nargs="+",
                        default=["Inclusion", "Patches", "Scratches"],
                        help="YOLO class names (order must match label class IDs)")
    parser.add_argument("--output_dir",  default="test_results",
                        help="Directory for YOLO val output")
    parser.add_argument("--save_csv",    default=None,
                        help="Optional path to save results as CSV")
    parser.add_argument("--device",      default=None,
                        help="Device override, e.g. 'cpu', '0' (auto-detected if omitted)")
    parser.add_argument("--conf",        type=float, default=None,
                        help="Confidence threshold for predictions "
                             "(default: Ultralytics val default of 0.001)")
    parser.add_argument("--iou",         type=float, default=None,
                        help="NMS IoU threshold, e.g. 0.4 "
                             "(default: Ultralytics val default of 0.7)")
    parser.add_argument("--per_class",   action="store_true",
                        help="Also report per-class mAP50 and mAP50-95")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.exists():
        sys.exit(f"Test directory not found: {test_dir}")

    display_names = args.names or args.model
    if len(display_names) != len(args.model):
        sys.exit("--names must have the same number of entries as --model")

    test_yaml = build_test_yaml(test_dir, args.class_names)
    print(f"Test set: {test_dir.resolve()}")
    print(f"Device:   {device}\n")

    rows = []
    for i, (model_path, name) in enumerate(zip(args.model, display_names)):
        out_dir = Path(args.output_dir) / Path(model_path).parent.parent.name
        print(f"[{i+1}/{len(args.model)}] Evaluating: {name}")
        row = eval_model(model_path, test_yaml, out_dir, device,
                         conf=args.conf, iou=args.iou, per_class=args.per_class)
        rows.append(row)
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

    print_table(rows, display_names)

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten per-class metrics into columns like "Inclusion_mAP50".
        flat_rows = []
        extra_cols = []
        for name, row in zip(display_names, rows):
            flat = {k: v for k, v in row.items() if k != "per_class"}
            for pc in row.get("per_class", []):
                for metric in ("mAP50", "mAP50-95"):
                    col = f"{pc['class']}_{metric}"
                    flat[col] = pc[metric]
                    if col not in extra_cols:
                        extra_cols.append(col)
            flat_rows.append({"name": name, **flat})

        base_cols = [k for k in rows[0].keys() if k != "per_class"]
        fieldnames = ["name"] + base_cols + extra_cols
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for flat in flat_rows:
                writer.writerow(flat)
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
