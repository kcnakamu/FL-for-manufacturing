"""
Diagnostic: ground-truth bounding-box overlap analysis.

For the TRAIN and VALIDATION splits of each federated client (test split is
deliberately left untouched), this computes pairwise IoU between ground-truth
boxes THAT SHARE THE SAME CLASS, within each image. Cross-class overlap is
intentionally ignored because NMS is class-wise by default (and only becomes
class-agnostic if agnostic=True), so overlap between boxes of different classes
never affects suppression.

It aggregates the IoU values into distributions:
    * overall (all clients, all classes pooled)
    * per class (pooled across clients)
    * per client (pooled across classes)
    * per client x class

and reports, for each distribution: number of pairs, mean, median,
90th/95th/99th percentile and max IoU. Every same-class pair with IoU > 0.5 is
flagged with its image filename for visual inspection. Histograms are written to
the output directory.

Usage:
    python utils/analysis/gt_box_overlap.py \
        --data_dir data/neu_data \
        --clients client_0 client_1 client_2 \
        --class_names Inclusion Patches Scratches \
        --output_dir utils/analysis/gt_box_overlap_out
"""

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from _gt_boxes import IMAGE_EXTS, load_gt_boxes

FLAG_THRESHOLD = 0.5


def xywh_to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    """IoU of two boxes given as (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def collect_pairs(data_dir: Path, clients, splits):
    """
    Walk every client/split, compute same-class pairwise IoU per image.

    Returns:
        pairs: list of dicts, one per same-class box pair, with keys
               client, split, cls, iou, image (relative path str).
        square_ok: True if every inspected image is square (so normalized-coord
               IoU equals pixel IoU); False if any non-square image was found.
        img_count, box_count: totals actually processed.
    """
    from PIL import Image

    pairs = []
    square_ok = True
    non_square_examples = []
    img_count = 0
    box_count = 0

    for client in clients:
        for split in splits:
            img_dir = data_dir / client / "images" / split
            label_dir = data_dir / client / "labels" / split
            if not img_dir.exists():
                print(f"  [skip] no image dir: {img_dir}")
                continue
            images = sorted(
                p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
            )
            for img_path in images:
                img_count += 1
                gt = load_gt_boxes(label_dir / f"{img_path.stem}.txt")
                box_count += len(gt)
                if len(gt) < 2:
                    continue

                # Verify the square assumption; scale to pixels if not square so
                # IoU is always computed in the true (pixel) aspect ratio.
                try:
                    W, H = Image.open(img_path).size
                except Exception:
                    W, H = 1, 1
                if W != H:
                    square_ok = False
                    if len(non_square_examples) < 5:
                        non_square_examples.append((img_path.name, W, H))

                # Group boxes by class, then pair within each class only.
                by_cls = defaultdict(list)
                for cls, cx, cy, w, h in gt:
                    by_cls[cls].append(xywh_to_xyxy(cx * W, cy * H, w * W, h * H))

                rel = str(img_path.relative_to(data_dir))
                for cls, boxes in by_cls.items():
                    for ba, bb in combinations(boxes, 2):
                        pairs.append({
                            "client": client,
                            "split": split,
                            "cls": cls,
                            "iou": iou(ba, bb),
                            "image": rel,
                        })

    if non_square_examples:
        print(f"  [warn] non-square images found (IoU scaled to pixels): "
              f"{non_square_examples}")
    return pairs, square_ok, img_count, box_count


def summarize(ious):
    import numpy as np

    a = np.asarray(ious, dtype=float)
    if a.size == 0:
        return None
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def fmt_row(label, s):
    if s is None:
        return f"{label:<28} {'--- no pairs ---':>50}"
    return (f"{label:<28} n={s['n']:>6}  mean={s['mean']:.4f}  "
            f"med={s['median']:.4f}  p90={s['p90']:.4f}  "
            f"p95={s['p95']:.4f}  p99={s['p99']:.4f}  max={s['max']:.4f}")


def plot_hist(ious, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not ious:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ious, bins=np.linspace(0, 1, 51), color="#1f77b4",
            edgecolor="white", linewidth=0.3)
    ax.axvline(FLAG_THRESHOLD, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"flag threshold = {FLAG_THRESHOLD}")
    ax.set_xlabel("IoU (same-class GT pairs)")
    ax.set_ylabel("count")
    ax.set_title(f"{title}  (n={len(ious)})", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Same-class ground-truth box overlap (IoU) diagnostic",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", default="data/neu_data",
                        help="Root containing <client>/images/<split> etc.")
    parser.add_argument("--clients", nargs="+",
                        default=["client_0", "client_1", "client_2"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Splits to analyze (test is intentionally excluded)")
    parser.add_argument("--class_names", nargs="+",
                        default=["Inclusion", "Patches", "Scratches"])
    parser.add_argument("--output_dir", default="utils/analysis/gt_box_overlap_out")
    args = parser.parse_args()

    if "test" in [s.lower() for s in args.splits]:
        sys.exit("Refusing to run: 'test' split must not be touched.")

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"data_dir not found: {data_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def cname(c):
        return args.class_names[c] if c < len(args.class_names) else f"class_{c}"

    print(f"Data dir: {data_dir.resolve()}")
    print(f"Clients:  {args.clients}")
    print(f"Splits:   {args.splits}  (test excluded)\n")

    pairs, square_ok, n_img, n_box = collect_pairs(
        data_dir, args.clients, args.splits)
    print(f"Processed {n_img} images, {n_box} GT boxes, "
          f"{len(pairs)} same-class pairs.")
    print(f"Square-image assumption held: {square_ok} "
          f"(if True, normalized IoU == pixel IoU exactly)\n")

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    # ---- Overall ----
    emit("=" * 100)
    emit("OVERALL (all clients, all classes pooled)")
    emit("=" * 100)
    emit(fmt_row("overall", summarize([p["iou"] for p in pairs])))
    emit("")

    # ---- Per class (pooled across clients) ----
    emit("=" * 100)
    emit("PER CLASS (pooled across clients)")
    emit("=" * 100)
    classes = sorted({p["cls"] for p in pairs})
    for c in classes:
        ious = [p["iou"] for p in pairs if p["cls"] == c]
        emit(fmt_row(cname(c), summarize(ious)))
    emit("")

    # ---- Per client (pooled across classes) ----
    emit("=" * 100)
    emit("PER CLIENT (pooled across classes)")
    emit("=" * 100)
    for client in args.clients:
        ious = [p["iou"] for p in pairs if p["client"] == client]
        emit(fmt_row(client, summarize(ious)))
    emit("")

    # ---- Per client x class ----
    emit("=" * 100)
    emit("PER CLIENT x CLASS")
    emit("=" * 100)
    for client in args.clients:
        for c in classes:
            ious = [p["iou"] for p in pairs
                    if p["client"] == client and p["cls"] == c]
            emit(fmt_row(f"{client} / {cname(c)}", summarize(ious)))
    emit("")

    # ---- Flagged pairs (IoU > threshold) ----
    flagged = sorted((p for p in pairs if p["iou"] > FLAG_THRESHOLD),
                     key=lambda p: p["iou"], reverse=True)
    emit("=" * 100)
    emit(f"FLAGGED PAIRS  (same-class IoU > {FLAG_THRESHOLD}):  {len(flagged)}")
    emit("=" * 100)
    if not flagged:
        emit("  none")
    else:
        emit(f"  {'IoU':>7}  {'client':<9}  {'split':<5}  {'class':<10}  image")
        for p in flagged:
            emit(f"  {p['iou']:>7.4f}  {p['client']:<9}  {p['split']:<5}  "
                 f"{cname(p['cls']):<10}  {p['image']}")
    emit("")

    # ---- Histograms ----
    plot_hist([p["iou"] for p in pairs], "Overall same-class GT IoU",
              out_dir / "hist_overall.png")
    for c in classes:
        plot_hist([p["iou"] for p in pairs if p["cls"] == c],
                  f"Same-class GT IoU - {cname(c)}",
                  out_dir / f"hist_class_{cname(c)}.png")
    for client in args.clients:
        plot_hist([p["iou"] for p in pairs if p["client"] == client],
                  f"Same-class GT IoU - {client}",
                  out_dir / f"hist_{client}.png")

    report_path = out_dir / "gt_box_overlap_report.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {report_path.resolve()}")
    print(f"Histograms written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
