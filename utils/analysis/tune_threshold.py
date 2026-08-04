"""
Sweep YOLO confidence thresholds and report the optimal value.

For each threshold the script runs model.val() and records mAP50, precision,
recall, and F1. The threshold that maximises --metric (default: f1) is
highlighted in the stdout table and written as "best_threshold" in the output
JSON.

Usage
-----
python utils/analysis/tune_threshold.py \
    --model  experiments/disruption_neu_fedavg/adaptation/neck_head/weights/best.pt \
    --data   data/neu_data/test/data.yaml \
    --split  test \
    --output experiments/disruption_neu_fedavg/analysis/threshold_sweep.json

python utils/analysis/tune_threshold.py \
    --model  experiments/disruption_neu_fedavg/fl/final_model/client_0_final.pt \
    --data   data/neu_data/client_2/data.yaml \
    --split  val \
    --metric map50 \
    --thresholds 0.1 0.9 0.05 \
    --output experiments/disruption_neu_fedavg/analysis/round10_threshold_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

_METRIC_KEYS = ("f1", "map50", "precision", "recall")


def _eval_at_threshold(model: YOLO, data: str, split: str, conf: float, device: str) -> dict:
    # plots=False: this sweep only needs the box metrics (mAP50/P/R/F1). The
    # confusion matrix — the only thing plots=True would additionally populate,
    # at the cost of writing plot files every threshold — is not used here.
    results = model.val(data=data, split=split, conf=conf, device=device, verbose=False, plots=False)

    mp    = float(results.box.mp)
    mr    = float(results.box.mr)
    map50 = float(results.box.map50)
    f1    = 2 * mp * mr / (mp + mr + 1e-8)

    return {
        "threshold": round(conf, 4),
        "map50":     round(map50, 4),
        "precision": round(mp, 4),
        "recall":    round(mr, 4),
        "f1":        round(f1, 4),
    }


def sweep(
    model_path: str,
    data: str,
    split: str,
    thresholds: list[float],
    metric: str,
    output: str,
    device: str,
) -> None:
    model = YOLO(model_path)
    print(f"Evaluating {len(thresholds)} thresholds ({split} split)…\n")

    results_list: list[dict] = []
    for conf in thresholds:
        row = _eval_at_threshold(model, data, split, conf, device)
        results_list.append(row)
        print(
            f"  conf={conf:.2f}  mAP50={row['map50']:.4f}  "
            f"P={row['precision']:.4f}  R={row['recall']:.4f}  "
            f"F1={row['f1']:.4f}"
        )

    best = max(results_list, key=lambda r: r[metric])

    print(f"\n{'='*60}")
    print(f"Best threshold by {metric}: {best['threshold']:.2f}  ({metric}={best[metric]:.4f})")
    print(f"  mAP50={best['map50']:.4f}  P={best['precision']:.4f}  "
          f"R={best['recall']:.4f}  F1={best['f1']:.4f}")
    print(f"{'='*60}\n")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_threshold": best["threshold"],
        "best_metric":    metric,
        "best_value":     best[metric],
        "model":          str(model_path),
        "data":           str(data),
        "split":          split,
        "sweep":          results_list,
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {output_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep YOLO confidence threshold on a dataset split.")
    parser.add_argument("--model",  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data",   required=True, help="Dataset YAML path")
    parser.add_argument("--split",  default="test", help="Dataset split to evaluate (test or val)")
    parser.add_argument(
        "--thresholds", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
        default=[0.1, 0.9, 0.05],
        help="Threshold range as START STOP STEP (inclusive). Default: 0.1 0.9 0.05",
    )
    parser.add_argument(
        "--metric", choices=_METRIC_KEYS, default="f1",
        help="Metric to maximise when selecting the best threshold.",
    )
    parser.add_argument(
        "--output", default="threshold_sweep.json",
        help="Output JSON path. Pass experiments/<exp_name>/analysis/threshold_sweep.json.",
    )
    parser.add_argument("--device", default="", help="Device (e.g. '0', 'cpu'). Empty = auto.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start, stop, step = args.thresholds
    thresholds = [round(v, 4) for v in np.arange(start, stop + step / 2, step)]
    sweep(
        model_path=args.model,
        data=args.data,
        split=args.split,
        thresholds=thresholds,
        metric=args.metric,
        output=args.output,
        device=args.device,
    )
