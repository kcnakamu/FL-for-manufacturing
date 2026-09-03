"""How much does the global model actually move between rounds?

A flat mAP curve has two very different causes and the metric alone cannot tell
them apart:

  * the global model is still being updated every round, but the updates no
    longer buy accuracy -- ordinary convergence.
  * the global model has stopped moving. Aggregation is returning (nearly) the
    same weights round after round, so the metric is flat because the *model* is
    frozen, not because it is converged.

This prints the relative step size ||g_t - g_{t-1}|| / ||g_{t-1}|| per round,
split by where in the network the movement is, so the two can be distinguished.

Usage:
    python scripts/diagnose_plateau.py experiments/fedavg_v4_seed1/fl
    python scripts/diagnose_plateau.py experiments/fedavg_v4_seed1/fl --compare experiments/fedavg_v4_seed0/fl
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


# YOLOv8 puts the Detect module last. Its cv3 branch produces the class logits
# and its cv2 branch the DFL box distribution; everything before it is the
# shared backbone + neck. Splitting them matters because a stalled *classifier*
# on a live backbone is a different failure from a globally frozen model.
def group_of(name: str, detect_idx: int) -> str:
    if f"model.{detect_idx}.cv3" in name:
        return "head_cls"
    if f"model.{detect_idx}.cv2" in name:
        return "head_box"
    if f"model.{detect_idx}." in name:
        return "head_other"
    return "backbone"


def load_round(path: Path) -> list[np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return [npz[f"arr_{i}"] for i in range(len(npz.files))]


def tensor_names(num_classes: int, expected: int) -> list[str] | None:
    """State-dict key order, which is the order shapley/logger.py serializes in.

    Returns None if a model cannot be built here (no torch, no weights) -- the
    script still reports the whole-model step size in that case.
    """
    try:
        from model import load_model, param_keys
        keys = param_keys(load_model(num_classes=num_classes))
    except Exception as exc:  # noqa: BLE001 - diagnostics must not hard-fail
        print(f"(could not build a reference model for tensor names: {exc})")
        print("(reporting whole-model step size only)\n")
        return None
    if len(keys) != expected:
        print(f"(reference model has {len(keys)} tensors, npz has {expected} -- "
              f"skipping the per-group split)\n")
        return None
    return keys


def rel_step(cur: list[np.ndarray], prev: list[np.ndarray],
             idx: list[int]) -> float:
    """||g_t - g_{t-1}|| / ||g_{t-1}|| over the selected tensors."""
    num = den = 0.0
    for i in idx:
        a, b = cur[i], prev[i]
        if not np.issubdtype(a.dtype, np.floating):
            continue  # num_batches_tracked and friends are counters, not weights
        d = a.astype(np.float64) - b.astype(np.float64)
        num += float(np.sum(d * d))
        den += float(np.sum(b.astype(np.float64) ** 2))
    if den == 0.0:
        return float("nan")
    return math.sqrt(num) / math.sqrt(den)


def map_curve(fl_dir: Path, clients: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for d in sorted(fl_dir.glob("round_*")):
        vals = []
        for cid in range(clients):
            m = d / f"client_{cid}_val" / "metrics.json"
            if m.exists():
                try:
                    v = json.loads(m.read_text()).get("mAP50")
                except json.JSONDecodeError:
                    continue
                if v is not None:
                    vals.append(v)
        if vals:
            out[int(re.sub(r"\D", "", d.name) or 0)] = sum(vals) / len(vals)
    return out


def analyze(fl_dir: Path, clients: int, num_classes: int, detect_idx: int):
    gdir = fl_dir / "globals"
    if not gdir.is_dir():
        raise SystemExit(f"no globals/ under {fl_dir} -- was the Shapley logger on?")
    rounds = sorted(int(p.stem.split("_")[-1]) for p in gdir.glob("global_round_*.npz"))
    if len(rounds) < 2:
        raise SystemExit(f"need at least 2 global snapshots, found {len(rounds)}")

    curve = map_curve(fl_dir, clients)
    prev = load_round(gdir / f"global_round_{rounds[0]:02d}.npz")
    names = tensor_names(num_classes, len(prev))
    groups = ["backbone", "head_box", "head_cls"]
    idx_all = list(range(len(prev)))
    idx_by_group = (
        {g: [i for i, n in enumerate(names) if group_of(n, detect_idx) == g]
         for g in groups}
        if names else {}
    )

    print(f"=== {fl_dir} ===")
    print(f"{len(prev)} tensors per snapshot, rounds {rounds[0]}..{rounds[-1]}\n")
    hdr = f"{'round':>6s} {'mAP50':>8s} {'step/|w|':>10s}"
    if idx_by_group:
        hdr += "".join(f"{g:>11s}" for g in groups)
    print(hdr)
    print("-" * len(hdr))

    steps: list[float] = []
    for r in rounds[1:]:
        cur = load_round(gdir / f"global_round_{r:02d}.npz")
        s = rel_step(cur, prev, idx_all)
        steps.append(s)
        m = curve.get(r)
        row = (f"{r:>6d} {m:>8.4f} " if m is not None else f"{r:>6d} {'--':>8s} ")
        row += f"{s:>10.2e}"
        for g in groups:
            if idx_by_group:
                row += f"{rel_step(cur, prev, idx_by_group[g]):>11.2e}"
        print(row)
        prev = cur
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fl_dir", help="experiments/<exp>/fl")
    ap.add_argument("--compare", help="a second run to print alongside")
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--num_classes", type=int, default=6)
    ap.add_argument("--detect_idx", type=int, default=22,
                    help="index of the Detect module in model.model (YOLOv8n: %(default)s)")
    args = ap.parse_args()

    a = analyze(Path(args.fl_dir), args.clients, args.num_classes, args.detect_idx)
    if args.compare:
        print()
        b = analyze(Path(args.compare), args.clients, args.num_classes, args.detect_idx)
        # Compare the tails, where a stalled run and a converging one diverge.
        tail = min(10, len(a), len(b))
        ma = sum(a[-tail:]) / tail
        mb = sum(b[-tail:]) / tail
        print(f"\nmean relative step over the last {tail} rounds:")
        print(f"  {Path(args.fl_dir).parent.name:<24s} {ma:.2e}")
        print(f"  {Path(args.compare).parent.name:<24s} {mb:.2e}")
        if mb > 0:
            print(f"  ratio: {ma / mb:.2f}x")


if __name__ == "__main__":
    main()
