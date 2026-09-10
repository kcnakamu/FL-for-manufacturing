"""Per-class AP@50 for several checkpoints, side by side.

check_fl_run.py and results.csv only carry aggregates, but the KD question is
about ONE class: whether distillation restores the knowledge the departed client
monopolised. A 0.12 aggregate gap could sit almost entirely on that class (KD has
a clear target) or be spread across the shared ones (it would not), and the
aggregate cannot tell those apart.

The first checkpoint named is the REFERENCE; every other column is reported as a
delta against it, so "federated vs centralized" and later "KD vs no-KD" read the
same way.

Usage:
    python scripts/compare_per_class.py \
        --data data/neu6s_centralized/data.yaml --split val \
        --out_dir experiments/analysis/neu6s_parity \
        centralized=experiments/baselines/centralized_neu6s/full/weights/best.pt \
        federated=experiments/pre_disruption_neu6s_seed0/fl/final_model/client_0_final.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shapley.evaluate import evaluate_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", metavar="NAME=PATH",
                    help="Checkpoints to compare; the first is the reference.")
    ap.add_argument("--data", required=True, help="Dataset YAML to evaluate against.")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out_dir", default="experiments/analysis/per_class")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    pairs = []
    for spec in args.models:
        if "=" not in spec:
            raise SystemExit(f"expected NAME=PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"{name}: no such checkpoint {p}")
        pairs.append((name, p))

    data = Path(args.data)
    class_names = [str(n) for n in yaml.safe_load(data.read_text())["names"]]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, p in pairs:
        print(f"[eval] {name}: {p} on {data} (split={args.split})")
        res = evaluate_checkpoint(
            p, str(data), device=args.device or None, imgsz=args.imgsz,
            out_dir=str((out_dir / "val_runs").resolve()), name=name,
            split=args.split,
        )
        per_class = res["per_class_ap50"]
        results[name] = {
            # Classes absent from ap_class_index were never scored -> 0.0.
            "per_class_ap50": {c: float(per_class.get(c, 0.0)) for c in class_names},
            "mAP50": float(res.get("mAP50", 0.0)),
            "mAP50-95": float(res.get("mAP50-95", 0.0)),
            "checkpoint": str(p),
        }

    ref = pairs[0][0]
    names = [n for n, _ in pairs]
    w = max(len(c) for c in class_names) + 2

    print("\n" + "=" * 78)
    print(f"PER-CLASS AP@50 on {data} [{args.split}]   reference = {ref}")
    print("=" * 78)
    header = f"{'class':{w}s}" + "".join(f"{n:>14s}" for n in names)
    header += "".join(f"{'d ' + n:>14s}" for n in names[1:])
    print(header); print("-" * len(header))
    for c in class_names:
        row = f"{c:{w}s}" + "".join(f"{results[n]['per_class_ap50'][c]:>14.4f}" for n in names)
        row += "".join(
            f"{results[n]['per_class_ap50'][c] - results[ref]['per_class_ap50'][c]:>+14.4f}"
            for n in names[1:]
        )
        print(row)
    print("-" * len(header))
    for key in ("mAP50", "mAP50-95"):
        row = f"{key:{w}s}" + "".join(f"{results[n][key]:>14.4f}" for n in names)
        row += "".join(f"{results[n][key] - results[ref][key]:>+14.4f}" for n in names[1:])
        print(row)

    out = out_dir / f"per_class_{args.split}.json"
    out.write_text(json.dumps({
        "data": str(data), "split": args.split, "reference": ref,
        "class_names": class_names, "results": results,
    }, indent=2))
    print(f"\n[DONE] -> {out}")


if __name__ == "__main__":
    main()
