"""Recover the aggregated global model from any federated round as a .pt.

The FL run saves only final_model/ at the end, but the Shapley logger persists
globals/global_round_XX.npz every round. t* is almost never the last round --
FedAvg curves plateau and then drift, so the best pre-disruption snapshot is
usually mid-run -- and this is how that snapshot becomes a usable checkpoint.

Usage:
    python scripts/extract_global.py \
        --log_dir experiments/pre_disruption_r40_seed0/fl/shapley_logs \
        --round 22 --out experiments/pre_disruption_r40_seed0/global_r22.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shapley.logger import load_global          # noqa: E402
from shapley.persistence import _save_pt        # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract one round's global model as a .pt")
    ap.add_argument("--log_dir", required=True, help="<exp>/fl/shapley_logs")
    ap.add_argument("--round", type=int, required=True, help="Round t to extract")
    ap.add_argument("--out", required=True, help="Destination .pt")
    ap.add_argument("--num_classes", type=int, default=6)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    npz = log_dir / "globals" / f"global_round_{args.round:02d}.npz"
    if not npz.exists():
        have = sorted(p.stem.split("_")[-1] for p in (log_dir / "globals").glob("*.npz"))
        raise SystemExit(
            f"no global for round {args.round} at {npz}\\n"
            f"rounds available: {', '.join(have) if have else '(none)'}"
        )

    arrays = load_global(log_dir, args.round)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _save_pt(arrays, args.num_classes, out)

    from ultralytics import YOLO
    nc = YOLO(str(out)).model.nc
    if nc != args.num_classes:
        raise SystemExit(f"wrote {out} but it has nc={nc}, expected {args.num_classes}")
    print(f"[OK] round {args.round} global -> {out}  (nc={nc})")
    print("     Note: class NAMES in a rebuilt checkpoint are placeholders; every "
          "evaluator here reads names from the dataset yaml, so this is expected.")


if __name__ == "__main__":
    main()
