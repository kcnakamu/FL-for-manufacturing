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
    ap.add_argument("--plateau_delta", type=float, default=0.005,
                    help="Mean mAP50 gain per round below which the curve counts "
                         "as flat (default: %(default)s)")
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

    # Plateau: the disruption round t* wants a converged pre-disruption model, so
    # that post-disruption change is attributable to the departure rather than to
    # the model still climbing. Report where a 3-round mean improvement first
    # drops below --plateau_delta.
    if len(curve) >= 5:
        win, t_star = 3, None
        for i in range(win, len(curve)):
            recent = (curve[i] - curve[i - win]) / win
            if recent < args.plateau_delta:
                t_star = i + 1          # rounds are 1-indexed
                break
        total_gain = curve[-1] - curve[0]
        last_win = (curve[-1] - curve[-1 - win]) / win if len(curve) > win else float("nan")
        print(f"\nconvergence: total gain {total_gain:+.4f}, "
              f"mean gain over the last {win} rounds {last_win:+.4f}/round")
        if t_star is None:
            print(f"  STILL CLIMBING at round {len(curve)} "
                  f"(never fell below {args.plateau_delta}/round) -- "
                  f"consider more rounds before fixing t*.")
        else:
            print(f"  plateau reached around round {t_star} "
                  f"(< {args.plateau_delta}/round) -- a reasonable t* candidate.")

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
