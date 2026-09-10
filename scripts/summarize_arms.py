"""Group KD ablation runs by arm and transfer set; report the departed class.

Evaluates each checkpoint ONCE and reports every metric from that pass, because
the headline needs three of them and they disagree: AP@50 integrates the whole
precision-recall curve, while recall at the operating threshold is what decides
whether a deployed model finds the class at all. On the departed class those
answer opposite questions, so a table showing only one is misleading.

Runs are discovered from the output-path convention written by run_distill.sh:
    <arm>_<transfer-set tag>_lam<L>_tc<T>_seed<S>
Seeds of the same configuration are aggregated as mean +/- sd.

Usage:
    python scripts/summarize_arms.py --root experiments/kd_neu6s \
        --class_name Scratches --device 0
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shapley.evaluate import evaluate_checkpoint  # noqa: E402

RUN_RE = re.compile(r"^(?P<arm>[a-z]+)_(?P<tag>.+?)_lam(?P<lam>[\d.]+)_tc(?P<tc>[\d.]+)_seed(?P<seed>\d+)$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="experiments/kd_neu6s")
    ap.add_argument("--data", default="data/neu6s_centralized/data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--class_name", default="Scratches")
    ap.add_argument("--out", default="experiments/analysis/arm_summary.json")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    a = ap.parse_args()

    runs = []
    for d in sorted(Path(a.root).iterdir()):
        m = RUN_RE.match(d.name)
        pt = d / "kd_neck_head" / "weights" / "best.pt"
        if m and pt.exists():
            runs.append((m.groupdict(), pt))
    if not runs:
        raise SystemExit(f"no runs matching the naming convention under {a.root}")
    print(f"[eval] {len(runs)} checkpoints, one pass each\n")

    rows = []
    for meta, pt in runs:
        r = evaluate_checkpoint(pt, a.data, device=a.device or None, imgsz=a.imgsz,
                                out_dir="experiments/analysis/_arm_val", name=pt.parent.parent.parent.name,
                                split=a.split)
        c = a.class_name
        rows.append({**meta, "ap": r["per_class_ap50"].get(c, 0.0),
                     "prec": r["per_class_precision"].get(c, 0.0),
                     "rec": r["per_class_recall"].get(c, 0.0),
                     "map50": r["mAP50"]})

    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["tag"], r["arm"]), []).append(r)

    def agg(vals):
        m = st.mean(vals)
        s = st.stdev(vals) if len(vals) > 1 else 0.0
        return m, s

    print(f"DEPARTED CLASS = {a.class_name}   ({a.split} split, mean +/- sd over seeds)")
    hdr = (f"{'transfer set':16s}{'arm':12s}{'n':>3s}{'recall':>16s}"
           f"{'precision':>16s}{'AP@50':>16s}{'mAP50':>16s}")
    print(hdr); print("-" * len(hdr))
    order = {"survivors": 0, "surv_buf10": 1, "surv_buf25": 2}
    arm_order = {"nokd": 0, "uniform": 1, "competence": 2}
    last = None
    for (tag, arm) in sorted(groups, key=lambda k: (order.get(k[0], 9), arm_order.get(k[1], 9))):
        g = groups[(tag, arm)]
        if last is not None and tag != last:
            print("-" * len(hdr))
        last = tag
        cells = []
        for key in ("rec", "prec", "ap", "map50"):
            m, s = agg([x[key] for x in g])
            cells.append(f"{m:.4f}±{s:.4f}")
        print(f"{tag:16s}{arm:12s}{len(g):>3d}" + "".join(f"{c:>16s}" for c in cells))
    print("-" * len(hdr))
    print("recall is at the operating threshold; AP@50 integrates the full curve.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"class": a.class_name, "split": a.split, "rows": rows}, indent=2))
    print(f"\n[DONE] -> {a.out}")


if __name__ == "__main__":
    main()
