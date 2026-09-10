"""Does a transfer set ever ELICIT a given class from the teacher?

Distillation moves only the knowledge a teacher demonstrates ON THE IMAGES IT IS
SHOWN. If the post-departure pool contains no instance of the departed class, a
competent teacher's correct answer on every batch is "not here" -- which is a
suppression signal, not a retention one, and weighting that teacher MORE
confidently makes it worse. This script measures that directly, so the claim is
checked rather than argued.

Reports, per image directory, how often the teacher puts any box of the target
class on the image and at what confidence.

Usage:
    python scripts/probe_transfer_set.py \
        --teacher experiments/teacher_bank_neu6s_seed0/teacher_bank/local_c5.pt \
        --class_name Scratches --device 0 \
        transfer=data/neu6s_survivors/images/train \
        departed=data/neu6s_data/client_4/images/train
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_CLASSES = ["Crazing", "Inclusion", "Patches",
                   "Pitted_surface", "Rolled-in_scale", "Scratches"]


def probe(model, img_dir: Path, cls_idx: int, n: int, imgsz: int, device: str):
    imgs = sorted(img_dir.glob("*.jpg"))
    if not imgs:
        raise SystemExit(f"no images in {img_dir}")
    random.Random(0).shuffle(imgs)
    imgs = imgs[:n]
    peaks = []
    for i in range(0, len(imgs), 16):
        res = model.predict([str(p) for p in imgs[i:i + 16]], imgsz=imgsz,
                            conf=0.001, verbose=False, device=device or None)
        for r in res:
            if r.boxes is None or len(r.boxes) == 0:
                peaks.append(0.0); continue
            cls = r.boxes.cls.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            hits = [c for k, c in zip(cls, cf) if int(k) == cls_idx]
            peaks.append(float(max(hits)) if hits else 0.0)
    peaks.sort()
    return peaks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", metavar="NAME=DIR")
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--class_name", required=True)
    ap.add_argument("--class_names", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--n", type=int, default=200, help="images sampled per directory")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25, help="threshold for the 'fires' rate")
    ap.add_argument("--device", default="")
    a = ap.parse_args()

    if a.class_name not in a.class_names:
        raise SystemExit(f"{a.class_name!r} not in {a.class_names}")
    idx = a.class_names.index(a.class_name)

    from ultralytics import YOLO
    model = YOLO(a.teacher)

    print(f"teacher : {a.teacher}")
    print(f"class   : {a.class_name} (index {idx})")
    print(f"{'':22s}{'images':>9s}{'fires >'+str(a.conf):>13s}{'median':>10s}{'p90':>10s}{'max':>10s}")
    print("-" * 74)
    for spec in a.dirs:
        if "=" not in spec:
            raise SystemExit(f"expected NAME=DIR, got {spec!r}")
        name, d = spec.split("=", 1)
        p = probe(model, Path(d), idx, a.n, a.imgsz, a.device)
        fires = sum(1 for v in p if v > a.conf)
        print(f"{name:22s}{len(p):>9d}{fires:>8d} ({100*fires/len(p):>4.1f}%)"
              f"{p[len(p)//2]:>10.4f}{p[int(len(p)*0.9)]:>10.4f}{p[-1]:>10.4f}")
    print("\nIf the transfer set's rate is ~0 while the departed set's is high, the")
    print("teacher never demonstrates the class during distillation: the KD term")
    print("carries only 'not here', and weighting that teacher harder suppresses more.")


if __name__ == "__main__":
    main()
