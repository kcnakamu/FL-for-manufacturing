"""What learning rate and training loss did the clients actually see, per round?

The aggregated global's step size decays geometrically in some runs -- a clean
constant factor per round, which is the signature of a schedule, not of noise.
Ultralytics writes lr/pg0 and the train losses per epoch into each round's
results.csv, so this reads them back and shows whether the learning rate is
being reset at the start of every round or is carrying over and annealing away.

Usage:
    python scripts/client_lr_trace.py experiments/fedavg_v4_seed1/fl
    python scripts/client_lr_trace.py experiments/fedavg_v4_seed1/fl --client 3
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def col(fieldnames: list[str], *needles: str) -> str | None:
    """Ultralytics pads its CSV headers with spaces and renames columns between
    versions, so match on substrings rather than exact keys."""
    for f in fieldnames:
        low = f.strip().lower()
        if all(n in low for n in needles):
            return f
    return None


def fnum(row: dict, key: str | None) -> float | None:
    if key is None:
        return None
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fl_dir", help="experiments/<exp>/fl")
    ap.add_argument("--client", type=int, default=0)
    args = ap.parse_args()

    fl = Path(args.fl_dir)
    rounds = sorted((d for d in fl.glob("round_*") if d.is_dir()),
                    key=lambda d: int(re.sub(r"\D", "", d.name) or 0))
    if not rounds:
        raise SystemExit(f"no round_* dirs under {fl}")

    print(f"=== {fl}  client {args.client} ===")
    print(f"{'round':>6s} {'epochs':>7s} {'lr first':>10s} {'lr last':>10s} "
          f"{'cls first':>10s} {'cls last':>10s} {'box last':>9s}")
    print("-" * 68)

    missing = 0
    for d in rounds:
        csv_path = d / f"client_{args.client}" / "results.csv"
        if not csv_path.exists():
            missing += 1
            continue
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            missing += 1
            continue
        names = list(rows[0].keys())
        k_lr = col(names, "lr/pg0") or col(names, "lr", "pg0")
        k_cls = col(names, "train", "cls_loss")
        k_box = col(names, "train", "box_loss")
        lr0, lr1 = fnum(rows[0], k_lr), fnum(rows[-1], k_lr)
        c0, c1 = fnum(rows[0], k_cls), fnum(rows[-1], k_cls)
        b1 = fnum(rows[-1], k_box)
        f = lambda v, w, p: (f"{v:>{w}.{p}e}" if v is not None else f"{'--':>{w}s}")
        g = lambda v, w: (f"{v:>{w}.4f}" if v is not None else f"{'--':>{w}s}")
        print(f"{d.name.replace('round_',''):>6s} {len(rows):>7d} "
              f"{f(lr0,10,3)} {f(lr1,10,3)} {g(c0,10)} {g(c1,10)} {g(b1,9)}")

    if missing:
        print(f"\n({missing} round(s) had no readable results.csv for client "
              f"{args.client})")
    print("\nIf 'lr first' is the same value every round, the schedule is being reset\n"
          "per round as intended. If it falls from round to round, the schedule is\n"
          "carrying over and the run is annealing itself to a standstill.")


if __name__ == "__main__":
    main()
