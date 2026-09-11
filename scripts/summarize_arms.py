"""Score every KD run on a held-out split and report the departed class.

Two problems with the first version of this table, both fixed here:

  * It scored on the 180-image validation set, which had already been used to
    derive the competence weights, pick each fine-tune's best.pt, and choose each
    model's recall threshold. --split test scores on the 270-image test set that
    nothing has touched.
  * Its "recall" came from Ultralytics' max-F1 operating point -- a threshold
    chosen per model, on the evaluation set itself (utils/metrics.py, the
    smooth(f1_curve.mean(0)).argmax() line). That is not what a deployed
    inspection line runs. --fixed_conf adds recall and precision at ONE fixed
    confidence for every model, computed by direct IoU>=0.5 matching, with the
    raw box counts reported so small-sample differences are visible.

Runs are discovered from the output-path convention written by run_distill.sh
and run_fedpost.sh:
    <arm>_<transfer-set tag>_lam<L>_tc<T>_seed<S>
Centralized fine-tunes are judged by their best.pt; federated runs by the final
aggregated global, which involves no validation-based selection at all.

Usage:
    python scripts/summarize_arms.py --split test --fixed_conf 0.25 --device 0 \
        --extra predeparture=experiments/pre_disruption_neu6s_seed0/fl/final_model/client_0_final.pt
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shapley.evaluate import evaluate_checkpoint  # noqa: E402

RUN_RE = re.compile(r"^(?P<arm>[a-z]+)_(?P<tag>.+?)_lam(?P<lam>[\d.]+)_tc(?P<tc>[\d.]+)_seed(?P<seed>\d+)$")
CKPTS = ("kd_neck_head/weights/best.pt",       # centralized fine-tune
         "fl/final_model/client_0_final.pt")   # federated: final aggregated global
TAG_ORDER = {"reference": -1, "survivors": 0, "surv_buf10": 1, "surv_buf25": 2, "fedsurv": 3}
ARM_ORDER = {"nokd": 0, "uniform": 1, "argmax": 2, "competence": 3}
PAIRS = [("competence", "nokd"), ("argmax", "nokd"), ("uniform", "nokd"),
         ("competence", "argmax"), ("competence", "uniform")]


def split_dirs(data_yaml: str, split: str):
    d = yaml.safe_load(Path(data_yaml).read_text())
    rel = d[split]
    return Path(d["path"]) / rel, Path(d["path"]) / rel.replace("images", "labels", 1)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def fixed_threshold(pt, img_dir: Path, lbl_dir: Path, cls_idx: int, conf: float,
                    imgsz: int, device: str, iou_thr: float = 0.5):
    """TP/FP/FN for one class at one fixed confidence, greedy IoU matching."""
    from ultralytics import YOLO
    model = YOLO(str(pt))
    imgs = sorted(img_dir.glob("*.jpg"))
    tp = fp = fn = 0
    for i in range(0, len(imgs), 16):
        chunk = imgs[i:i + 16]
        res = model.predict([str(p) for p in chunk], imgsz=imgsz, conf=conf,
                            verbose=False, device=device or None)
        for p, r in zip(chunk, res):
            h, w = r.orig_shape
            gts = []
            lf = lbl_dir / f"{p.stem}.txt"
            if lf.exists():
                for line in lf.read_text().splitlines():
                    s = line.split()
                    if s and int(float(s[0])) == cls_idx:
                        cx, cy, bw, bh = map(float, s[1:5])
                        gts.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                                    (cx + bw / 2) * w, (cy + bh / 2) * h))
            preds = []
            if r.boxes is not None and len(r.boxes):
                for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                   r.boxes.cls.cpu().numpy()):
                    if int(k) == cls_idx:
                        preds.append((float(c), tuple(float(v) for v in b)))
            preds.sort(key=lambda t: -t[0])
            used = [False] * len(gts)
            for _, b in preds:
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    if not used[j]:
                        v = _iou(b, g)
                        if v > best:
                            best, bj = v, j
                if best >= iou_thr:
                    used[bj] = True
                    tp += 1
                else:
                    fp += 1
            fn += used.count(False)
    return tp, fp, fn


def _ms(vals):
    m = st.mean(vals)
    return m, (st.stdev(vals) if len(vals) > 1 else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="experiments/kd_neu6s")
    ap.add_argument("--data", default="data/neu6s_centralized/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--fixed_conf", type=float, default=0.25)
    ap.add_argument("--class_name", default="Scratches")
    ap.add_argument("--extra", nargs="*", default=[], metavar="NAME=PATH",
                    help="Single reference checkpoints, e.g. the pre-departure global.")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="Only score these arms -- keeps a rescore from picking up a "
                         "run still mid-training, whose best.pt is a partial model.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    a = ap.parse_args()

    names = [str(n) for n in yaml.safe_load(Path(a.data).read_text())["names"]]
    cls_idx = names.index(a.class_name)
    img_dir, lbl_dir = split_dirs(a.data, a.split)

    runs = []
    for spec in a.extra:
        n, p = spec.split("=", 1)
        runs.append(({"arm": n, "tag": "reference", "seed": "0"}, Path(p)))
    for d in sorted(Path(a.root).iterdir()):
        m = RUN_RE.match(d.name)
        if not m:
            continue
        if a.arms and m.group("arm") not in a.arms:
            continue
        pt = next((d / c for c in CKPTS if (d / c).exists()), None)
        if pt is not None:
            runs.append((m.groupdict(), pt))
    print(f"[eval] {len(runs)} checkpoints on the {a.split} split, one pass each\n")

    rows = []
    for meta, pt in runs:
        r = evaluate_checkpoint(pt, a.data, device=a.device or None, imgsz=a.imgsz,
                                out_dir="experiments/analysis/_arm_eval",
                                name=f"{meta['tag']}_{meta['arm']}_s{meta['seed']}", split=a.split)
        tp, fp, fn = fixed_threshold(pt, img_dir, lbl_dir, cls_idx, a.fixed_conf, a.imgsz, a.device)
        c = a.class_name
        rows.append({**meta, "ckpt": str(pt),
                     "tp": tp, "fp": fp, "fn": fn,
                     "rec_fix": tp / (tp + fn) if tp + fn else 0.0,
                     "prec_fix": tp / (tp + fp) if tp + fp else float("nan"),
                     "ap": r["per_class_ap50"].get(c, 0.0),
                     "rec_f1": r["per_class_recall"].get(c, 0.0),
                     "map50": r["mAP50"]})

    n_gt = rows[0]["tp"] + rows[0]["fn"] if rows else 0
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["tag"], r["arm"]), []).append(r)

    print(f"DEPARTED CLASS = {a.class_name} · {a.split} split · {n_gt} ground-truth boxes · "
          f"mean ± sd over seeds")
    hdr = (f"{'condition':14s}{'arm':13s}{'n':>3s}{'rec@'+str(a.fixed_conf):>17s}{'boxes':>10s}"
           f"{'prec@'+str(a.fixed_conf):>17s}{'AP@50':>17s}{'rec@maxF1':>17s}{'mAP50':>17s}")
    print(hdr); print("-" * len(hdr))
    last = None
    for key in sorted(groups, key=lambda k: (TAG_ORDER.get(k[0], 9), ARM_ORDER.get(k[1], 9), k[1])):
        g = groups[key]
        if last is not None and key[0] != last:
            print("-" * len(hdr))
        last = key[0]
        cells = []
        for f in ("rec_fix", "prec_fix", "ap", "rec_f1", "map50"):
            vals = [x[f] for x in g if x[f] == x[f]]
            cells.append(f"{_ms(vals)[0]:.4f}±{_ms(vals)[1]:.4f}" if vals else "—")
        tps = sum(x["tp"] for x in g) / len(g)
        print(f"{key[0]:14s}{key[1]:13s}{len(g):>3d}{cells[0]:>17s}{f'{tps:.1f}/{n_gt}':>10s}"
              + "".join(f"{c:>17s}" for c in cells[1:]))
    print("-" * len(hdr))

    print(f"\nPAIRED per seed (same condition, same seed) · departed class · |d|/sd > 3 = decisive")
    for tag in sorted({k[0] for k in groups if k[0] != "reference"}, key=lambda t: TAG_ORDER.get(t, 9)):
        for x, y in PAIRS:
            A = {r["seed"]: r for r in groups.get((tag, x), [])}
            B = {r["seed"]: r for r in groups.get((tag, y), [])}
            seeds = sorted(set(A) & set(B))
            if not seeds:
                continue
            out = []
            for f, lbl in (("rec_fix", f"rec@{a.fixed_conf}"), ("ap", "AP@50")):
                d = [A[s][f] - B[s][f] for s in seeds]
                m, sd = _ms(d)
                ratio = abs(m) / sd if sd > 0 else float("inf")
                verd = "decisive" if ratio > 3 else ("suggestive" if ratio > 1.5 else "unresolved")
                out.append(f"{lbl} {m:+.4f}±{sd:.4f} ({verd})")
            print(f"  {tag:12s} {x:>10s} − {y:<10s} n={len(seeds)}   " + "   ".join(out))

    out = Path(a.out or f"experiments/analysis/arm_summary_{a.split}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"class": a.class_name, "split": a.split, "fixed_conf": a.fixed_conf,
                               "n_gt": n_gt, "rows": rows}, indent=2))
    print(f"\n[DONE] -> {out}")


if __name__ == "__main__":
    main()
