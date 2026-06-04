"""
Evaluate a trained YOLO model on the global test set.

Usage:
    # single model
    python utils/evaluate_test.py --model fl_runs/20240101_fedavg/round_10/client_0/weights/best.pt

    # multiple models (prints a comparison table)
    python utils/evaluate_test.py --model a/weights/best.pt b/weights/best.pt --names A B

    # custom data dir and save CSV
    python utils/evaluate_test.py --model best.pt --data_dir data_aug --save_csv results.csv
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


def eval_model(model_path: str, test_yaml: str, output_dir: Path, device: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=test_yaml,
        split="test",
        imgsz=480,
        batch=16,
        workers=0,
        verbose=False,
        device=device,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    p  = float(metrics.box.mp)
    r  = float(metrics.box.mr)
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {
        "model":    model_path,
        "P":        p,
        "R":        r,
        "F1":       f1,
        "mAP50":    float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
    }


def print_table(rows: list[dict], names: list[str]):
    header = f"{'Name':<30}  {'P':>7}  {'R':>7}  {'F1':>7}  {'mAP50':>7}  {'mAP50-95':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name, row in zip(names, rows):
        print(
            f"{name:<30}  {row['P']:>7.4f}  {row['R']:>7.4f}  {row['F1']:>7.4f}"
            f"  {row['mAP50']:>7.4f}  {row['mAP50-95']:>9.4f}"
        )
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
    parser.add_argument("--class_names", nargs="+", default=["surface_crack"],
                        help="YOLO class names")
    parser.add_argument("--output_dir",  default="test_results",
                        help="Directory for YOLO val output")
    parser.add_argument("--save_csv",    default=None,
                        help="Optional path to save results as CSV")
    parser.add_argument("--device",      default=None,
                        help="Device override, e.g. 'cpu', '0' (auto-detected if omitted)")
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
        row = eval_model(model_path, test_yaml, out_dir, device)
        rows.append(row)
        print(
            f"  P={row['P']:.4f}  R={row['R']:.4f}  F1={row['F1']:.4f}"
            f"  mAP50={row['mAP50']:.4f}  mAP50-95={row['mAP50-95']:.4f}"
        )

    Path(test_yaml).unlink(missing_ok=True)

    print_table(rows, display_names)

    if args.save_csv:
        csv_path = Path(args.save_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name"] + list(rows[0].keys()))
            writer.writeheader()
            for name, row in zip(display_names, rows):
                writer.writerow({"name": name, **row})
        print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
