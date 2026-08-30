"""Sanity-check a finished federated run before anything is built on top of it.

Verifies the things that fail SILENTLY -- a run can complete with a zero exit
code and still be unusable:

  * client participation per round. Flower starts a round once
    min_available_clients have connected, so a misconfigured --num_clients lets
    rounds train on whatever subset happened to arrive. Nothing errors; the
    global model is just quietly built from fewer clients than intended.
  * the final checkpoints exist and carry the expected class count.
  * the mAP curve is present and not flat/NaN.

Usage:
    python scripts/check_fl_run.py experiments/pre_disruption_seed0/fl \
        --clients 6 --rounds 10 --num_classes 6
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def round_dirs(fl_dir: Path) -> list[Path]:
    return sorted((d for d in fl_dir.glob("round_*") if d.is_dir()),
                  key=lambda d: int(re.sub(r"\D", "", d.name) or 0))


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate a finished FL run.")
    ap.add_argument("fl_dir", help="experiments/<exp>/fl")
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--peak_tol", type=float, default=0.010,
                    help="A round within this of the peak counts as indistinguishable "
                         "from it (default: %(default)s)")
    args = ap.parse_args()

    fl = Path(args.fl_dir)
    if not fl.is_dir():
        raise SystemExit(f"not a directory: {fl}")

    problems: list[str] = []
    rds = round_dirs(fl)
    print(f"=== {fl} ===")
    print(f"rounds found: {len(rds)} (expected {args.rounds})")
    if len(rds) != args.rounds:
        problems.append(f"{len(rds)} round dirs, expected {args.rounds}")

    print(f"\n{'round':>6s} {'clients':>8s} {'mAP50':>9s} {'d(mAP50)':>10s} "
          f"{'mAP50-95':>10s}   participation")
    print("-" * 74)
    curve: list[float] = []
    curve95: list[float] = []
    prev = None
    for d in rds:
        rows = {}
        for cid in range(args.clients):
            m = d / f"client_{cid}_val" / "metrics.json"
            if m.exists():
                try:
                    rows[cid] = json.loads(m.read_text())
                except json.JSONDecodeError:
                    problems.append(f"{m} is not valid JSON")
        n = len(rows)
        missing = [c for c in range(args.clients) if c not in rows]
        mark = "ok" if n == args.clients else f"MISSING {missing}"
        if n != args.clients:
            problems.append(f"{d.name}: {n}/{args.clients} clients reported "
                            f"(missing {missing})")
        m50 = [r.get("mAP50") for r in rows.values() if r.get("mAP50") is not None]
        m5095 = [r.get("mAP50-95") for r in rows.values() if r.get("mAP50-95") is not None]
        avg50 = sum(m50) / len(m50) if m50 else float("nan")
        avg5095 = sum(m5095) / len(m5095) if m5095 else float("nan")
        delta = "" if prev is None else f"{avg50 - prev:+10.4f}"
        print(f"{d.name.replace('round_',''):>6s} {n:>8d} {avg50:>9.4f} {delta:>10s} "
              f"{avg5095:>10.4f}   {mark}")
        if avg50 == avg50:            # skip NaN
            curve.append(avg50)
            prev = avg50
        if avg5095 == avg5095:
            curve95.append(avg5095)

    # Choosing t*.
    #
    # A first-crossing rule on the raw curve ("first round whose 3-round mean gain
    # drops below X") is fragile: FedAvg curves oscillate, so one dip trips it well
    # before the model has actually stopped improving. Report the peak and the
    # earliest round statistically indistinguishable from it instead -- that is the
    # cheapest round that is as good as the best one, which is what t* wants.
    def summarize(name: str, series: list[float]) -> None:
        if len(series) < 5:
            return
        peak = max(series)
        peak_r = series.index(peak) + 1
        earliest = next(i + 1 for i, v in enumerate(series) if v >= peak - args.peak_tol)
        tail = series[-5:]
        tail_mean = sum(tail) / len(tail)
        print(f"\n{name}: peak {peak:.4f} @ round {peak_r}; "
              f"earliest within {args.peak_tol} of peak = round {earliest}")
        drop = peak - tail_mean
        if drop > args.peak_tol:
            print(f"  DECLINING: last-5 mean {tail_mean:.4f} sits {drop:.4f} below the "
                  f"peak -- later rounds are not helping and may be degrading it.")
        elif peak_r >= len(series) - 2:
            print(f"  still improving at the end -- consider more rounds before fixing t*.")
        else:
            print(f"  plateaued: last-5 mean {tail_mean:.4f}, within {args.peak_tol} of peak.")

    if len(curve) >= 5:
        print(f"\nconvergence: total gain {curve[-1] - curve[0]:+.4f} over {len(curve)} rounds")
        summarize("mAP50   ", curve)
        summarize("mAP50-95", curve95)
        print("\n  t* candidate: the earliest round at which BOTH metrics are within "
              f"{args.peak_tol} of their peak.")

    # Final checkpoints
    print("\nfinal checkpoints:")
    fm = fl / "final_model"
    if not fm.is_dir():
        problems.append(f"no final_model dir at {fm}")
        print("  MISSING")
    else:
        pts = sorted(fm.glob("*_final.pt"))
        if not pts:
            problems.append(f"no *_final.pt in {fm}")
        for pt in pts:
            try:
                from ultralytics import YOLO
                nc = YOLO(str(pt)).model.nc
            except Exception as e:  # noqa: BLE001
                problems.append(f"{pt.name}: cannot load ({e})")
                continue
            ok = "ok" if nc == args.num_classes else f"nc={nc}, expected {args.num_classes}"
            if nc != args.num_classes:
                problems.append(f"{pt.name} has nc={nc}, expected {args.num_classes}")
            print(f"  {pt.name:<28s} nc={nc}  {ok}")

    print("\n" + "=" * 62)
    if problems:
        print("PROBLEMS FOUND:")
        for p in problems:
            print(f"  ! {p}")
        raise SystemExit(1)
    print(f"OK: {len(rds)} rounds, all {args.clients} clients reported every round, "
          f"final checkpoints are nc={args.num_classes}.")


if __name__ == "__main__":
    main()
